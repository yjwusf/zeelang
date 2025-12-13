//===- StableHLOToZee.cpp - StableHLO to Zee conversion -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion from StableHLO dialect to Zee dialect.
//
//===----------------------------------------------------------------------===//

#include "Zee/Conversions/Passes.h"
#include "Zee/ZeeDialect.h"
#include "Zee/ZeeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace zee {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOZEE
#include "Zee/Conversions/Passes.cpp.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class StableHLOToZeeTypeConverter : public TypeConverter {
public:
  StableHLOToZeeTypeConverter() {
    // Pass through all types unchanged
    addConversion([](Type type) { return type; });
  }
};

//===----------------------------------------------------------------------===//
// Binary Elementwise Op Conversions
//===----------------------------------------------------------------------===//

template <typename StableHLOOp, typename ZeeOp>
class BinaryElementwiseOpConversion : public OpConversionPattern<StableHLOOp> {
public:
  using OpConversionPattern<StableHLOOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StableHLOOp op, typename StableHLOOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ZeeOp>(op, op.getType(), adaptor.getLhs(),
                                        adaptor.getRhs());
    return success();
  }
};

using ConvertAddOp =
    BinaryElementwiseOpConversion<stablehlo::AddOp, zee::AddOp>;
using ConvertSubtractOp =
    BinaryElementwiseOpConversion<stablehlo::SubtractOp, zee::SubtractOp>;
using ConvertMulOp =
    BinaryElementwiseOpConversion<stablehlo::MulOp, zee::MultiplyOp>;
using ConvertDivOp =
    BinaryElementwiseOpConversion<stablehlo::DivOp, zee::DivideOp>;
using ConvertMaxOp =
    BinaryElementwiseOpConversion<stablehlo::MaxOp, zee::MaximumOp>;
using ConvertMinOp =
    BinaryElementwiseOpConversion<stablehlo::MinOp, zee::MinimumOp>;

//===----------------------------------------------------------------------===//
// Unary Elementwise Op Conversions
//===----------------------------------------------------------------------===//

template <typename StableHLOOp, typename ZeeOp>
class UnaryElementwiseOpConversion : public OpConversionPattern<StableHLOOp> {
public:
  using OpConversionPattern<StableHLOOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StableHLOOp op, typename StableHLOOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ZeeOp>(op, op.getType(), adaptor.getOperand());
    return success();
  }
};

using ConvertAbsOp =
    UnaryElementwiseOpConversion<stablehlo::AbsOp, zee::AbsOp>;
using ConvertNegOp =
    UnaryElementwiseOpConversion<stablehlo::NegOp, zee::NegateOp>;
using ConvertExpOp =
    UnaryElementwiseOpConversion<stablehlo::ExpOp, zee::ExpOp>;
using ConvertLogOp =
    UnaryElementwiseOpConversion<stablehlo::LogOp, zee::LogOp>;

//===----------------------------------------------------------------------===//
// Constant Op Conversion
//===----------------------------------------------------------------------===//

class ConvertConstantOp : public OpConversionPattern<stablehlo::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::ConstantOp op,
                  stablehlo::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<zee::ConstantOp>(op, op.getType(),
                                                  op.getValueAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class ConvertStableHLOToZeePass
    : public impl::ConvertStableHLOToZeeBase<ConvertStableHLOToZeePass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Mark Zee dialect as legal
    target.addLegalDialect<zee::ZeeDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // Mark StableHLO ops that we convert as illegal
    target.addIllegalOp<stablehlo::AddOp, stablehlo::SubtractOp,
                        stablehlo::MulOp, stablehlo::DivOp, stablehlo::MaxOp,
                        stablehlo::MinOp, stablehlo::AbsOp, stablehlo::NegOp,
                        stablehlo::ExpOp, stablehlo::LogOp,
                        stablehlo::ConstantOp>();

    // Setup type converter
    StableHLOToZeeTypeConverter typeConverter;

    // Populate conversion patterns
    RewritePatternSet patterns(context);
    patterns.add<ConvertAddOp, ConvertSubtractOp, ConvertMulOp, ConvertDivOp,
                 ConvertMaxOp, ConvertMinOp, ConvertAbsOp, ConvertNegOp,
                 ConvertExpOp, ConvertLogOp, ConvertConstantOp>(typeConverter,
                                                                 context);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace zee
} // namespace mlir
