//===- ZeeOps.td - Zee dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZEE_OPS
#define ZEE_OPS

include "Zee/ZeeDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Zee_FooOp : Zee_Op<"foo", [Pure,
                               SameOperandsAndResultType]> {
  let summary = "Illustrates how to define an operation.";
  let description = [{
    The `zee.foo` operation illustrates how to define a new
    operation in a dialect. It uses an integer argument and
    returns an integer.

    Example:

    ```mlir
    %0 = arith.constant 2 : i32
    // Apply the foo operation to %0
    %1 = zee.foo %0 : i32
    ```
  }];

  let arguments = (ins I32:$input);
  let results = (outs I32:$res);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Tensor type constraint for elementwise ops
//===----------------------------------------------------------------------===//

def Zee_Tensor : AnyTypeOf<[AnyRankedTensor]>;

//===----------------------------------------------------------------------===//
// Binary elementwise operations (compatible with StableHLO)
//===----------------------------------------------------------------------===//

class Zee_BinaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Zee_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Zee_Tensor:$lhs, Zee_Tensor:$rhs);
  let results = (outs Zee_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def Zee_AddOp : Zee_BinaryElementwiseOp<"add"> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two tensors.

    Example:
    ```mlir
    %result = zee.add %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def Zee_SubtractOp : Zee_BinaryElementwiseOp<"subtract"> {
  let summary = "Element-wise subtraction operation";
  let description = [{
    Performs element-wise subtraction of two tensors.

    Example:
    ```mlir
    %result = zee.subtract %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def Zee_MultiplyOp : Zee_BinaryElementwiseOp<"multiply"> {
  let summary = "Element-wise multiplication operation";
  let description = [{
    Performs element-wise multiplication of two tensors.

    Example:
    ```mlir
    %result = zee.multiply %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def Zee_DivideOp : Zee_BinaryElementwiseOp<"divide"> {
  let summary = "Element-wise division operation";
  let description = [{
    Performs element-wise division of two tensors.

    Example:
    ```mlir
    %result = zee.divide %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def Zee_MaximumOp : Zee_BinaryElementwiseOp<"maximum"> {
  let summary = "Element-wise maximum operation";
  let description = [{
    Computes element-wise maximum of two tensors.

    Example:
    ```mlir
    %result = zee.maximum %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def Zee_MinimumOp : Zee_BinaryElementwiseOp<"minimum"> {
  let summary = "Element-wise minimum operation";
  let description = [{
    Computes element-wise minimum of two tensors.

    Example:
    ```mlir
    %result = zee.minimum %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Unary elementwise operations
//===----------------------------------------------------------------------===//

class Zee_UnaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Zee_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Zee_Tensor:$operand);
  let results = (outs Zee_Tensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($result)
  }];
}

def Zee_AbsOp : Zee_UnaryElementwiseOp<"abs"> {
  let summary = "Element-wise absolute value operation";
  let description = [{
    Computes element-wise absolute value.

    Example:
    ```mlir
    %result = zee.abs %operand : tensor<4xf32>
    ```
  }];
}

def Zee_NegateOp : Zee_UnaryElementwiseOp<"negate"> {
  let summary = "Element-wise negation operation";
  let description = [{
    Computes element-wise negation.

    Example:
    ```mlir
    %result = zee.negate %operand : tensor<4xf32>
    ```
  }];
}

def Zee_ExpOp : Zee_UnaryElementwiseOp<"exponential"> {
  let summary = "Element-wise exponential operation";
  let description = [{
    Computes element-wise exponential (e^x).

    Example:
    ```mlir
    %result = zee.exponential %operand : tensor<4xf32>
    ```
  }];
}

def Zee_LogOp : Zee_UnaryElementwiseOp<"log"> {
  let summary = "Element-wise natural logarithm operation";
  let description = [{
    Computes element-wise natural logarithm.

    Example:
    ```mlir
    %result = zee.log %operand : tensor<4xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Constant operation
//===----------------------------------------------------------------------===//

def Zee_ConstantOp : Zee_Op<"constant", [Pure]> {
  let summary = "Constant tensor operation";
  let description = [{
    Produces a constant tensor from a constant value.

    Example:
    ```mlir
    %result = zee.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs Zee_Tensor:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Print operation (for JIT demonstration)
//===----------------------------------------------------------------------===//

def Zee_PrintOp : Zee_Op<"print"> {
  let summary = "Print an integer value";
  let description = [{
    Prints an integer value to stdout. This operation is used for
    demonstrating JIT execution.

    Example:
    ```mlir
    %c42 = arith.constant 42 : i32
    zee.print %c42 : i32
    ```
  }];

  let arguments = (ins I32:$input);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

#endif // ZEE_OPS
