//===- zee-opt.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Zee/ZeeDialect.h"
#include "Zee/ZeePasses.h"
#include "Zee/Conversions/Passes.h"

#ifdef ZEE_ENABLE_STABLEHLO
#include "stablehlo/dialect/Register.h"
#endif

int main(int argc, char **argv) {
  // Register zee passes.
  mlir::zee::registerPasses();

  // Register ZeeToArith conversion pass (always available).
  mlir::zee::registerConvertZeeToArith();

#ifdef ZEE_ENABLE_STABLEHLO
  // Register StableHLO conversion pass.
  mlir::zee::registerConvertStableHLOToZee();
#endif

  mlir::DialectRegistry registry;
  registry.insert<mlir::zee::ZeeDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

#ifdef ZEE_ENABLE_STABLEHLO
  // Register StableHLO dialects (stablehlo, chlo, vhlo)
  mlir::stablehlo::registerAllDialects(registry);
#endif

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Zee optimizer driver\n", registry));
}
