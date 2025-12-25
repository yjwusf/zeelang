//===- Passes.td - Zee conversion passes -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZEE_CONVERSION_PASSES
#define ZEE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertStableHLOToZee : Pass<"convert-stablehlo-to-zee"> {
  let summary = "Convert StableHLO dialect to Zee dialect";
  let description = [{
    This pass converts StableHLO operations to equivalent operations in the
    Zee dialect. This enables using StableHLO as an input format for the
    Zee compilation pipeline.

    Example:
    ```mlir
    // Input (StableHLO):
    func.func @add(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.add %arg0, %arg1 : tensor<4xi32>
      return %0 : tensor<4xi32>
    }

    // Output (Zee):
    func.func @add(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
      %0 = zee.add %arg0, %arg1 : tensor<4xi32>
      return %0 : tensor<4xi32>
    }
    ```
  }];

  let dependentDialects = [
    "mlir::zee::ZeeDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
  ];
}

def ConvertZeeToArith : Pass<"convert-zee-to-arith"> {
  let summary = "Convert Zee dialect to Arith dialect";
  let description = [{
    This pass lowers Zee dialect operations to equivalent operations in the
    Arith dialect. For example, zee.foo is lowered to arith.addi (treating
    it as an addition with itself).

    Example:
    ```mlir
    // Input (Zee):
    func.func @test(%arg0: i32) -> i32 {
      %0 = zee.foo %arg0 : i32
      return %0 : i32
    }

    // Output (Arith):
    func.func @test(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      return %0 : i32
    }
    ```
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
  ];
}

def LowerToLLVM : Pass<"lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower all dialects to LLVM dialect for JIT execution";
  let description = [{
    This pass performs full lowering of high-level dialects (Arith, Func, SCF,
    Affine, MemRef) to the LLVM dialect. This enables JIT compilation and
    execution using the MLIR ExecutionEngine.

    The lowering happens in multiple stages through transitive lowering:
    - Affine -> Standard
    - SCF -> ControlFlow
    - Arith -> LLVM
    - MemRef -> LLVM
    - Func -> LLVM
    - ControlFlow -> LLVM

    Example:
    ```mlir
    // Input:
    func.func @main() -> i32 {
      %c42 = arith.constant 42 : i32
      return %c42 : i32
    }

    // Output (LLVM):
    llvm.func @main() -> i32 {
      %0 = llvm.mlir.constant(42 : i32) : i32
      llvm.return %0 : i32
    }
    ```
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif // ZEE_CONVERSION_PASSES
