//===- ZeeTypes.td - Zee dialect types ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZEE_TYPES
#define ZEE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Zee/ZeeDialect.td"

//===----------------------------------------------------------------------===//
// Zee type definitions
//===----------------------------------------------------------------------===//

class Zee_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Zee_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Zee_CustomType : Zee_Type<"Custom", "custom"> {
  let summary = "Zee custom type";
  let description = "Custom type in zee dialect";
  let parameters = (ins StringRefParameter<"the custom value">:$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // ZEE_TYPES
