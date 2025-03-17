// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rematerialize-parallel-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REMATERIALIZEPARALLELOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool isScalarOrTensorOfSizeOne(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return tensorType.hasStaticShape() && tensorType.getNumElements() == 1;
  }
  return t.isIntOrIndexOrFloat();
}

/// Rematerialize all parallel elementwise operations into its users within a
/// `flow.dispatch.region`.
struct RematerializeParallelOpsPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Avoid doing this for scalar operations.
    auto isScalarValue = [](Value v) {
      return isScalarOrTensorOfSizeOne(v.getType());
    };
    if (llvm::all_of(genericOp.getOperands(), isScalarValue) &&
        llvm::all_of(genericOp.getResults(), isScalarValue)) {
      return failure();
    }

    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!linalg::areElementwiseOpsFusable(&opOperand))
        continue;

      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusionResult)) {
        auto replacements = fusionResult->fusedOp->getResults().take_back(
            genericOp.getNumResults());
        // Copy over any non native attributes for the operation.
        auto prunedAttributeList = linalg::getPrunedAttributeList(genericOp);
        fusionResult->fusedOp->setAttrs(prunedAttributeList);
        rewriter.replaceOp(genericOp, replacements);
        return success();
      }
    }
    return failure();
  }
};

// TODO: Fusability of such producers into single-input LinalgExt ops can be
// generalized as an interface. Even further generalization would be possible
// if we actually started implementing LinalgFusionInterface for the dialect
// ops.
struct FuseSimpleEltwiseProducerIntoScanPattern
    : public OpRewritePattern<IREE::LinalgExt::ScanOp> {
  using OpRewritePattern<IREE::LinalgExt::ScanOp>::OpRewritePattern;

private:
  using IREEExtScanOp = IREE::LinalgExt::ScanOp;

  // TODO: linalg.elemwise_unary ops are known to satisfy the main criteria -
  // should we support them as well by generating a region ad hoc based on
  // 'fun' & 'cast' attribute values?
  FailureOr<linalg::GenericOp> getSimpleEltwiseProducer(IREEExtScanOp op) const {
    if (failed(op.verify())) {
      return failure();
    }

    OpOperand *input = op.getDpsInputOperand(0);
    assert(input && "Expected a valid input operand at index 0");
    auto producer = input->get().getDefiningOp<linalg::GenericOp>();
    if (!producer)
      return failure();
    if (producer.getNumResults() != 1 || producer.getNumDpsInputs() != 1)
      return failure();
    if (!producer.isAllParallelLoops())
      return failure();
    if (producer.hasIndexSemantics())
      return failure();
    // After initial sanity checks, we can be confident that fully parallel
    // bit extend/truncate ops satisafy the criteria.
    if (IREE::LinalgExt::isBitExtendOp(producer) ||
        IREE::LinalgExt::isBitTruncateOp(producer))
      return producer;

    // Yet to check matching shapes, AffineMap triviality, probably single
    // region

    return producer;
  }

public:
  LogicalResult matchAndRewrite(IREE::LinalgExt::ScanOp scanOp,
                                PatternRewriter &rewriter) const override {
    auto maybeProducer = getSimpleEltwiseProducer(scanOp);
    if (failed(maybeProducer))
      return failure();
    linalg::GenericOp producer = maybeProducer.value();

    auto fusedScanOp = rewriter.create<IREE::LinalgExt::ScanOp>(
        scanOp.getLoc(), scanOp.getResultTypes(), producer.getDpsInputs(),
        scanOp.getOutputs(), scanOp.getDimensionAttr(),
        scanOp.getInclusiveAttr());

    Block *fusedBlock = rewriter.createBlock(&fusedScanOp.getRegion());

    Block &producerBlock = producer->getRegion(0).front();
    Block &scanBlock = scanOp->getRegion(0).front();
    OpBuilder::InsertionGuard guard(rewriter);
    IRMapping mapper;

    assert(scanBlock.getNumArguments() == 2 &&
           "Expected linalg_ext.scan input block to have 2 arguments");
    auto producerInputArg = producerBlock.getArgument(0);
    auto producerResultOp =
        cast<linalg::YieldOp>(producerBlock.getTerminator())->getOperand(0);

    auto processedScanArg = scanBlock.getArgument(0);
    mapper.map(processedScanArg, fusedBlock->insertArgument(
                                     (unsigned)0, processedScanArg.getType(),
                                     processedScanArg.getLoc()));

    mapper.map(producerInputArg,
               fusedBlock->insertArgument(1, producerInputArg.getType(),
                                          producerInputArg.getLoc()));
    for (auto &producerOp : producerBlock.without_terminator()) {
      rewriter.clone(producerOp, mapper);
    }
    mapper.map(scanBlock.getArgument(1),
               mapper.lookupOrDefault(producerResultOp));
    for (auto &consumerOp : scanBlock) {
      rewriter.clone(consumerOp, mapper);
    }

    rewriter.replaceOp(scanOp, fusedScanOp);
    // TODO: Figure out if attribute pruning of any kind is needed
    return success();
  }
};

struct RematerializeParallelOpsPass final
    : impl::RematerializeParallelOpsPassBase<RematerializeParallelOpsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet fusionPatterns(funcOp.getContext());
    fusionPatterns.insert<RematerializeParallelOpsPattern>(funcOp.getContext());
    fusionPatterns.insert<FuseSimpleEltwiseProducerIntoScanPattern>(
        funcOp.getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
