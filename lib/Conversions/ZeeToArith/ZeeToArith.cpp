//===- ZeeToArith.cpp - Zee to Arith conversion -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion from Zee dialect to Arith dialect.
//
//===----------------------------------------------------------------------===//

#include "Zee/ZeeDialect.h"
#include "Zee/ZeeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace zee {

#define GEN_PASS_DEF_CONVERTZEETOARITH
#include "Zee/Conversions/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// FooOp Conversion - lower zee.foo to arith.addi (add input to itself)
//===----------------------------------------------------------------------===//

class ConvertFooOp : public OpConversionPattern<zee::FooOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(zee::FooOp op, zee::FooOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower zee.foo %x to arith.addi %x, %x (treat foo as "add to self")
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getInput(),
                                                adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertZeeToArithPass
    : public impl::ConvertZeeToArithBase<ConvertZeeToArithPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Mark Arith and Func dialects as legal
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // Mark zee.foo as illegal (must be converted)
    target.addIllegalOp<zee::FooOp>();

    // Populate conversion patterns
    RewritePatternSet patterns(context);
    patterns.add<ConvertFooOp>(context);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace zee
} // namespace mlir
