// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SplitReduction.cpp ----------------------------===//
//
// Split reduction dimension to increase parallelism of a linalg operation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SPLITREDUCTIONPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t>
    splitReductionRatio("iree-dispatch-creation-split-matmul-reduction",
                        llvm::cl::desc("split ratio"), llvm::cl::init(1));

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-dispatch-creation-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

static LogicalResult splitReductionOnMatmul(
    RewriterBase &rewriter, linalg::MatmulOp op,
    linalg::ControlSplitReductionFn controlSplitReductionFn) {
  // Since user information about compilation are passed through attributes we
  // need to make sure to propagate those.
  SmallVector<NamedAttribute> prunedAttributeList =
      linalg::getPrunedAttributeList(op);

  FailureOr<linalg::SplitReductionResult> result =
      linalg::splitReduction(rewriter, op, controlSplitReductionFn);
  if (failed(result)) {
    return failure();
  }

  result->splitLinalgOp->setAttrs(prunedAttributeList);
  return result;
}

/*struct ArgMaxSplitReductionPattern*/
/*    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {*/
/*  ArgMaxSplitReductionPattern(mlir::MLIRContext *ctx,*/
/*                              int64_t chunkSize = 2004)*/
/*      : OpRewritePattern(ctx), chunkSize(chunkSize) {}*/
/**/
/*  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,*/
/*                                      mlir::PatternRewriter &rewriter) const*/
/*      override {*/
/*    // --- 1. Quick structural match ----------------------------------------*/
/*    if (op.getNumLoops() != 1 ||                       // 1‑D only*/
/*        !linalg::isReductionIterator(op.getIteratorTypesArray()[0]) || */
/*        op.getNumDpsInits() != 2)                       // value + index*/
/*      return mlir::failure();*/
/**/
/*    auto inTy  = mlir::cast<mlir::RankedTensorType>(op.getDpsInputOperand(0)->get().getType());*/
/*    auto valTy = mlir::cast<mlir::RankedTensorType>(op.getDpsInitOperand(0)->get().getType());*/
/*    auto idxTy = mlir::cast<mlir::RankedTensorType>(op.getDpsInitOperand(1)->get().getType());*/
/*    if (!inTy || !inTy.hasStaticShape() || inTy.getRank() != 1 ||*/
/*        !valTy || !idxTy)*/
/*      return mlir::failure();*/
/**/
/*    int64_t nElems = inTy.getDimSize(0);*/
/*    if (nElems % chunkSize != 0)*/
/*      return mlir::failure();*/
/*    int64_t nChunks = nElems / chunkSize;*/
/**/
/*    // --- 2. IR building helpers -------------------------------------------*/
/*    auto loc      = op.getLoc();*/
/*    auto f16Ty    = valTy.getElementType();*/
/*    auto i32Ty    = idxTy.getElementType();*/
/*    auto vals1Ty  = mlir::RankedTensorType::get({nChunks}, f16Ty);*/
/*    auto idxs1Ty  = mlir::RankedTensorType::get({nChunks}, i32Ty);*/
/**/
/*    auto zeroF16  = rewriter.create<mlir::arith::ConstantOp>(*/
/*                       loc, f16Ty, rewriter.getFloatAttr(f16Ty, 0.0));*/
/*    auto zeroI32  = rewriter.create<mlir::arith::ConstantOp>(*/
/*                       loc, i32Ty, rewriter.getI32IntegerAttr(0));*/
/*    auto cChunkSz = rewriter.create<mlir::arith::ConstantIndexOp>(loc, chunkSize);*/
/**/
/*    // --- 3. Allocate / fill intermediate tensors --------------------------*/
/*    auto initVals1 = rewriter*/
/*        .create<mlir::tensor::EmptyOp>(loc, vals1Ty.getShape(), f16Ty)*/
/*        .getResult();*/
/*    auto initIdxs1 = rewriter*/
/*        .create<mlir::tensor::EmptyOp>(loc, idxs1Ty.getShape(), i32Ty)*/
/*        .getResult();*/
/*    auto filledVals1 =*/
/*        rewriter.create<mlir::linalg::FillOp>(loc, zeroF16, initVals1)*/
/*            .getResult(0);*/
/*    auto filledIdxs1 =*/
/*        rewriter.create<mlir::linalg::FillOp>(loc, zeroI32, initIdxs1)*/
/*            .getResult(0);*/
/**/
/*    // --- 4. Reshape input to [nChunks x chunkSize] -------------------------*/
/*    auto reshapedTy =*/
/*        mlir::RankedTensorType::get({nChunks, chunkSize}, inTy.getElementType());*/
/*    auto reshaped =*/
/*        rewriter.create<mlir::tensor::ExpandShapeOp>(*/
/*            loc, reshapedTy, op.getInputs()[0],*/
/*            mlir::ReassociationIndices{{0, 1}})*/
/*            .getResult();*/
/**/
/*    // --- 5. First‑stage parallel reduction --------------------------------*/
/*    auto maps1 = rewriter.getAffineMapArrayAttr(*/
/*        {mlir::AffineMap::getMultiDimIdentityMap(2, getContext()),*/
///*         mlir::AffineMap::get(/*dimCount=*/2, /*symCount=*/0,*/
/*                              {rewriter.getAffineDimExpr(0)}),*/
///*         mlir::AffineMap::get(/*dimCount=*/2, /*symCount=*/0,*/
/*                              {rewriter.getAffineDimExpr(0)})});*/
/*    auto iterTypes1 = rewriter.getStrArrayAttr({"parallel", "reduction"});*/
/**/
/*    auto firstStage = rewriter.create<mlir::linalg::GenericOp>(*/
/*        loc, mlir::TypeRange{vals1Ty, idxs1Ty},*/
///*        /*inputs=*/reshaped,*/
///*        /*outputs=*/mlir::ValueRange{filledVals1, filledIdxs1},*/
/*        maps1, iterTypes1,*/
///*        /*bodyBuild=*/*/
/*        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {*/
/*          mlir::Value in = args[0], curVal = args[1], curIdx = args[2];*/
/**/
/*          // truncate to f16*/
/*          auto inF16 = b.create<mlir::arith::TruncFOp>(loc, f16Ty, in);*/
/*          // idx = chunkIdx * chunkSize + localIdx*/
/*          auto lIdx   = b.create<mlir::linalg::IndexOp>(loc, 1);*/
/*          auto cIdx   = b.create<mlir::linalg::IndexOp>(loc, 0);*/
/*          auto base   = b.create<mlir::arith::MulIOp>(loc, cIdx, cChunkSz);*/
/*          auto gIdx   = b.create<mlir::arith::AddIOp>(loc, base, lIdx);*/
/*          auto gIdx32 = b.create<mlir::arith::IndexCastOp>(loc, i32Ty, gIdx);*/
/**/
/*          auto cmp    = b.create<mlir::arith::CmpFOp>(*/
/*              loc, mlir::arith::CmpFPredicate::OGT, inF16, curVal);*/
/*          auto newVal = b.create<mlir::arith::SelectOp>(loc, cmp, inF16, curVal);*/
/*          auto newIdx = b.create<mlir::arith::SelectOp>(loc, cmp, gIdx32,*/
/*                                                        curIdx);*/
/*          b.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{newVal, newIdx});*/
/*        });*/
/**/
/*    // --- 6. Second‑stage reduction over the 64 partials --------------------*/
/*    auto outValInit = rewriter.create<mlir::tensor::EmptyOp>(*/
/*                          loc, mlir::RankedTensorType::get({}, f16Ty))*/
/*                          .getResult();*/
/*    auto outIdxInit = rewriter.create<mlir::tensor::EmptyOp>(*/
/*                          loc, mlir::RankedTensorType::get({}, i32Ty))*/
/*                          .getResult();*/
/*    auto valInit =*/
/*        rewriter.create<mlir::linalg::FillOp>(loc, zeroF16, outValInit)*/
/*            .getResult(0);*/
/*    auto idxInit =*/
/*        rewriter.create<mlir::linalg::FillOp>(loc, zeroI32, outIdxInit)*/
/*            .getResult(0);*/
/**/
///*    /*auto maps2 = rewriter.getAffineMapArrayAttr(*/*/
///*    /*    {rewriter.getDimIdentityMap(), rewriter.getDimIdentityMap(),*/*/
///*    /*     rewriter.getDimIdentityMap(), rewriter.getDimIdentityMap()});*/*/
/*    auto maps2 = rewriter.getAffineMapArrayAttr({*/
/*        mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext()),*/
/*        mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext()),*/
/*        mlir::AffineMap::get(1, 0, {}, rewriter.getContext()),*/
/*        mlir::AffineMap::get(1, 0, {}, rewriter.getContext())*/
/*    });*/
/*    auto iterTypes2 = rewriter.getStrArrayAttr({"reduction"});*/
/**/
/*    auto secondStage = rewriter.create<mlir::linalg::GenericOp>(*/
/*        loc, mlir::TypeRange{valTy, idxTy},*/
///*        /*inputs=*/mlir::ValueRange{firstStage.getResult(0),*/
/*                                    firstStage.getResult(1)},*/
///*        /*outputs=*/mlir::ValueRange{valInit, idxInit},*/
/*        maps2, iterTypes2,*/
/*        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {*/
/*          mlir::Value inVal = args[0], inIdx = args[1],*/
/*                      curVal = args[2], curIdx = args[3];*/
/**/
/*          auto cmp = b.create<mlir::arith::CmpFOp>(*/
/*              loc, mlir::arith::CmpFPredicate::OGT, inVal, curVal);*/
/*          auto newVal =*/
/*              b.create<mlir::arith::SelectOp>(loc, cmp, inVal, curVal);*/
/*          auto newIdx =*/
/*              b.create<mlir::arith::SelectOp>(loc, cmp, inIdx, curIdx);*/
/*          b.create<mlir::linalg::YieldOp>(loc,*/
/*                                          mlir::ValueRange{newVal, newIdx});*/
/*        });*/
/**/
/*    // --- 7. Replace original op -------------------------------------------*/
/*    rewriter.replaceOp(op, secondStage->getResults());*/
/*    return mlir::success();*/
/*  }*/
/**/
/*private:*/
/*  int64_t chunkSize;*/
/*};*/


namespace {
struct SplitReductionPass final
    : public impl::SplitReductionPassBase<SplitReductionPass> {
  void runOnOperation() override {
    if (splitReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty()) {
      return;
    }

    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ArgMaxSplitReductionPattern>(context, 2004);
    mlir::applyOpPatternsGreedily(funcOp, std::move(patterns));

    auto matmulSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      // For matmul make the new parallel dimension first so that it looks
      // like a batch_matmul and can follow the same codegen.
      return {int64_t(splitReductionRatio), 0, /*innerParallel=*/false};
    };

    SmallVector<linalg::MatmulOp> matmulCandidates;
    IRRewriter rewriter(context);
    funcOp->walk([&](linalg::MatmulOp op) { matmulCandidates.push_back(op); });
    for (auto op : matmulCandidates) {
      (void)splitReductionOnMatmul(rewriter, op, matmulSplitReductionControlFn);
    }

    IREE::LinalgExt::TopkSplitReductionControlFn topkSplitReductionControlFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),
                                           topkSplitReductionRatio.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };

    SmallVector<IREE::LinalgExt::TopkOp> topkCandidates;
    funcOp->walk(
        [&](IREE::LinalgExt::TopkOp op) { topkCandidates.push_back(op); });
    for (auto op : topkCandidates) {
      (void)splitReduction(rewriter, op, topkSplitReductionControlFn);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
