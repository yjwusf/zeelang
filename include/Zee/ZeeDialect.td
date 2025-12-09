//===- ZeeDialect.td - Zee dialect -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZEE_DIALECT
#define ZEE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Zee dialect definition.
//===----------------------------------------------------------------------===//

def Zee_Dialect : Dialect {
  let name = "zee";
  let summary = "A zee out-of-tree MLIR dialect.";
  let description = [{
    This dialect is an example of an out-of-tree MLIR dialect designed to
    illustrate the basic setup required to develop MLIR-based tools without
    working inside of the LLVM source tree.
  }];
  let cppNamespace = "::mlir::zee";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base zee operation definition.
//===----------------------------------------------------------------------===//

class Zee_Op<string mnemonic, list<Trait> traits = []> :
        Op<Zee_Dialect, mnemonic, traits>;

#endif // ZEE_DIALECT
