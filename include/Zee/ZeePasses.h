//===- ZeePasses.h - Zee passes ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ZEE_ZEEPASSES_H
#define ZEE_ZEEPASSES_H

#include "Zee/ZeeDialect.h"
#include "Zee/ZeeOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace zee {
#define GEN_PASS_DECL
#include "Zee/ZeePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Zee/ZeePasses.h.inc"
} // namespace zee
} // namespace mlir

#endif // ZEE_ZEEPASSES_H
