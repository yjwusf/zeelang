//===- ZeeDialect.cpp - Zee dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Zee/ZeeDialect.h"
#include "Zee/ZeeOps.h"
#include "Zee/ZeeTypes.h"

using namespace mlir;
using namespace mlir::zee;

#include "Zee/ZeeOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Zee dialect.
//===----------------------------------------------------------------------===//

void ZeeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Zee/ZeeOps.cpp.inc"
      >();
  registerTypes();
}
