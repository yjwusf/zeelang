//===- zee-jit.cpp - Zee JIT Compiler and Runner ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Zee JIT compiler, which takes MLIR input with Zee
// dialect operations, lowers them through Arith to LLVM, and executes the
// result using the MLIR ExecutionEngine.
//
// Based on MLIR Toy Tutorial Chapter 6.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "Zee/ZeeDialect.h"
#include "Zee/ZeePasses.h"
#include "Zee/Conversions/Passes.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<bool> enableOpt("O",
                                      llvm::cl::desc("Enable optimizations"),
                                      llvm::cl::init(false));

static llvm::cl::opt<bool>
    dumpLLVMIR("dump-llvm-ir",
               llvm::cl::desc("Dump the LLVM IR after lowering"),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    dumpMLIR("dump-mlir",
             llvm::cl::desc("Dump the MLIR after each lowering stage"),
             llvm::cl::init(false));

/// Load and parse the input MLIR file.
static OwningOpRef<ModuleOp> loadAndParseModule(MLIRContext &context,
                                                  StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

/// Run the lowering pipeline: zee -> arith -> llvm
static LogicalResult lowerToLLVM(ModuleOp module, bool dumpIntermediate) {
  PassManager pm(module->getContext());

  // Apply any generic pass manager command line options.
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  // Add the lowering passes
  // First: lower Zee dialect to Arith dialect
  pm.addPass(zee::createConvertZeeToArith());

  if (dumpIntermediate) {
    llvm::errs() << "=== After Zee to Arith lowering ===\n";
  }

  // Second: lower everything to LLVM dialect
  pm.addPass(zee::createLowerToLLVM());

  if (failed(pm.run(module)))
    return failure();

  if (dumpIntermediate) {
    llvm::errs() << "=== Final LLVM Dialect ===\n";
    module.print(llvm::errs());
    llvm::errs() << "\n";
  }

  return success();
}

/// JIT compile and execute the module.
static int runJit(ModuleOp module, bool optimize, bool dumpIR) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = makeOptimizingTransformer(
      optimize ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;

  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Failed to construct an execution engine\n";
    llvm::handleAllErrors(maybeEngine.takeError(),
                          [](const llvm::ErrorInfoBase &info) {
                            llvm::errs() << "Error: " << info.message() << "\n";
                          });
    return 1;
  }

  auto &engine = maybeEngine.get();

  // Optionally dump the LLVM IR.
  if (dumpIR) {
    llvm::errs() << "=== LLVM IR ===\n";
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (llvmModule) {
      llvmModule->print(llvm::errs(), nullptr);
    }
    llvm::errs() << "\n";
  }

  // Look up the 'main' function and invoke it directly.
  auto mainFn = engine->lookup("main");
  if (!mainFn) {
    llvm::errs() << "Failed to find 'main' function\n";
    llvm::handleAllErrors(mainFn.takeError(),
                          [](const llvm::ErrorInfoBase &info) {
                            llvm::errs() << "Error: " << info.message() << "\n";
                          });
    return 1;
  }

  // Cast the function pointer and call it.
  auto fn = (int32_t (*)())mainFn.get();
  int32_t result = fn();
  return result;
}

int main(int argc, char **argv) {
  // Register pass command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "Zee JIT Compiler\n");

  MLIRContext context;
  // Load all required dialects
  context.getOrLoadDialect<zee::ZeeDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  // Load and parse the module.
  auto module = loadAndParseModule(context, inputFilename);
  if (!module) {
    llvm::errs() << "Failed to parse module\n";
    return 1;
  }

  if (dumpMLIR) {
    llvm::errs() << "=== Input MLIR ===\n";
    module->print(llvm::errs());
    llvm::errs() << "\n";
  }

  // Lower to LLVM dialect.
  if (failed(lowerToLLVM(*module, dumpMLIR))) {
    llvm::errs() << "Lowering to LLVM dialect failed\n";
    return 1;
  }

  // JIT compile and run.
  return runJit(*module, enableOpt, dumpLLVMIR);
}
