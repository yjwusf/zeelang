//===- ZeeOps.cpp - Zee dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Zee/ZeeOps.h"
#include "Zee/ZeeDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::zee;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

#define GET_OP_CLASSES
#include "Zee/ZeeOps.cpp.inc"
