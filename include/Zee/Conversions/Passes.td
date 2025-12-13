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

#endif // ZEE_CONVERSION_PASSES
