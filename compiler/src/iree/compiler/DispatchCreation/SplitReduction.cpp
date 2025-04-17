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


#define DEBUG_TYPE "split-reduction"


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

struct ArgMaxSplitReductionPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  ArgMaxSplitReductionPattern(mlir::MLIRContext *ctx, int64_t chunkSize = 2004)
      : OpRewritePattern(ctx), chunkSize(chunkSize) {}

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter &rewriter) const
      override {
    LLVM_DEBUG(llvm::dbgs() << "Trying to apply ArgMaxSplitReductionPattern to:\n" << op << "\n");

    // --- 1. Quick structural match ----------------------------------------
    if (op.getNumLoops() != 1 || // 1‑D reduction only
        !mlir::linalg::isReductionIterator(op.getIteratorTypesArray()[0]) ||
        op.getNumDpsInits() != 2) // Expecting 2 outputs: value + index
      return mlir::failure();

    // Check the body for the typical argmax structure (optional but good practice)
    // - A comparison (e.g., arith.cmpf)
    // - Two selects using the comparison result
    // - Yielding the selected value and index
    // (This part is omitted for brevity but recommended for robustness)


    // --- 2. Type checking and shape analysis -----------------------------
    if (op.getNumDpsInputs() != 1) return mlir::failure(); // Expect 1 input

    auto inputOperand = op.getDpsInputOperand(0);
    auto valInitOperand = op.getDpsInitOperand(0);
    auto idxInitOperand = op.getDpsInitOperand(1);

    if (!inputOperand || !valInitOperand || !idxInitOperand) return mlir::failure();

    auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(inputOperand->get().getType());
    auto valTy = mlir::dyn_cast<mlir::RankedTensorType>(valInitOperand->get().getType()); // Output value type
    auto idxTy = mlir::dyn_cast<mlir::RankedTensorType>(idxInitOperand->get().getType()); // Output index type

    if (!inTy || !inTy.hasStaticShape() || inTy.getRank() != 1 ||
        !valTy || valTy.getRank() != 0 || // Expect scalar output value tensor
        !idxTy || idxTy.getRank() != 0)   // Expect scalar output index tensor
      return mlir::failure();

    // Ensure index type is an integer type
    mlir::Type idxElemTy = idxTy.getElementType();
    if (!mlir::isa<mlir::IntegerType>(idxElemTy)) {
        LLVM_DEBUG(llvm::dbgs() << "Index type is not an integer type: " << idxElemTy << "\n");
        return mlir::failure();
    }

    // Ensure value type is a float type (adapt if needed for integers)
    mlir::Type valElemTy = valTy.getElementType();
     if (!mlir::isa<mlir::FloatType>(valElemTy)) {
         LLVM_DEBUG(llvm::dbgs() << "Value type is not a float type: " << valElemTy << "\n");
         // Note: Could extend this to support integer argmax if required.
         return mlir::failure();
     }

    int64_t nElems = inTy.getDimSize(0);
    if (chunkSize <= 0 || nElems <= chunkSize || nElems % chunkSize != 0) {
        LLVM_DEBUG(llvm::dbgs() << "Input size " << nElems << " not divisible by chunk size " << chunkSize << " or chunk size invalid\n");
        return mlir::failure(); // Require divisibility for this simple split
    }
    int64_t nChunks = nElems / chunkSize;

    // --- 3. IR building helpers -------------------------------------------
    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    // Use the actual element types derived from the original op
    auto vals1Ty = mlir::RankedTensorType::get({nChunks}, valElemTy);
    auto idxs1Ty = mlir::RankedTensorType::get({nChunks}, idxElemTy);

    // Create zero constants with the *correct* types
    mlir::FloatAttr zeroValAttr;
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(valElemTy)) {
        zeroValAttr = rewriter.getFloatAttr(valElemTy, 0.0);
    } else {
         // Handle other potential value types if necessary
         op.emitError("Unsupported value type for zero initialization");
         return mlir::failure();
    }
    mlir::Value zeroVal = rewriter.create<mlir::arith::ConstantOp>(loc, valElemTy, zeroValAttr);

    // Use getIntegerAttr for arbitrary integer types
    mlir::Value zeroIdx = rewriter.create<mlir::arith::ConstantOp>(
        loc, idxElemTy, rewriter.getIntegerAttr(idxElemTy, 0));

    auto cChunkSz = rewriter.create<mlir::arith::ConstantIndexOp>(loc, chunkSize);

    // --- 4. Allocate / fill intermediate tensors --------------------------
    auto initVals1 = rewriter
        .create<mlir::tensor::EmptyOp>(loc, vals1Ty.getShape(), valElemTy)
        .getResult();
    auto initIdxs1 = rewriter
        .create<mlir::tensor::EmptyOp>(loc, idxs1Ty.getShape(), idxElemTy)
        .getResult();
    auto filledVals1 =
        rewriter.create<mlir::linalg::FillOp>(loc, zeroVal, initVals1)
            .getResult(0);
    auto filledIdxs1 =
        rewriter.create<mlir::linalg::FillOp>(loc, zeroIdx, initIdxs1)
            .getResult(0);

    // --- 5. Reshape input to [nChunks x chunkSize] -------------------------
    auto reshapedTy =
        mlir::RankedTensorType::get({nChunks, chunkSize}, inTy.getElementType());
    // Use the actual input operand value
    auto reshaped =
        rewriter.create<mlir::tensor::ExpandShapeOp>(
            loc, reshapedTy, inputOperand->get(),
            mlir::ReassociationIndices{{0, 1}})
            .getResult();

    // --- 6. First‑stage parallel reduction --------------------------------
    // Input: (d0, d1) -> (d0, d1)  [nChunks, chunkSize] -> [nChunks, chunkSize]
    // Output Val: (d0, d1) -> (d0) [nChunks, chunkSize] -> [nChunks]
    // Output Idx: (d0, d1) -> (d0) [nChunks, chunkSize] -> [nChunks]
    auto mapIn = mlir::AffineMap::getMultiDimIdentityMap(2, ctx);
    auto mapOut = mlir::AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)}, ctx);
    auto maps1 = llvm::SmallVector<mlir::AffineMap, 3>{mapIn, mapOut, mapOut};

    // Reduce along dimension 1 (chunkSize), keep dimension 0 (nChunks) parallel
    auto iterTypes1 = llvm::SmallVector<mlir::utils::IteratorType, 2>{
        mlir::utils::IteratorType::parallel, // Iterate over chunks
        mlir::utils::IteratorType::reduction // Reduce within a chunk
    };

    auto firstStage = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/mlir::TypeRange{vals1Ty, idxs1Ty},
        /*inputs=*/reshaped,
        /*outputs=*/mlir::ValueRange{filledVals1, filledIdxs1},
        maps1, iterTypes1,
        /*bodyBuild=*/
        [&](mlir::OpBuilder &b, mlir::Location bodyLoc, mlir::ValueRange args) {
          // args: input_element, current_max_val, current_max_idx
          mlir::Value inElem = args[0]; // Type from reshaped input (e.g., f32)
          mlir::Value curVal = args[1]; // Type valElemTy (e.g., f16)
          mlir::Value curIdx = args[2]; // Type idxElemTy (e.g., i64)

          // Cast/truncate input element to the comparison/storage type (valElemTy)
          mlir::Value inValProcessed;
          if (inElem.getType() == valElemTy) {
              inValProcessed = inElem;
          } else if (auto inFloatTy = mlir::dyn_cast<mlir::FloatType>(inElem.getType())) {
              if (auto valFloatTy = mlir::dyn_cast<mlir::FloatType>(valElemTy)) {
                  // Example: f32 -> f16 truncation
                  if (inFloatTy.getWidth() > valFloatTy.getWidth()) {
                      inValProcessed = b.create<mlir::arith::TruncFOp>(bodyLoc, valElemTy, inElem);
                  }
                  // Example: f16 -> f32 extension (less likely for argmax)
                  else if (inFloatTy.getWidth() < valFloatTy.getWidth()) {
                     inValProcessed = b.create<mlir::arith::ExtFOp>(bodyLoc, valElemTy, inElem);
                  } else {
                     inValProcessed = inElem; // Same float type
                  }
              } else {
                   // Cannot convert float input to non-float valElemTy easily here
                   op.emitError("Unsupported type combination for input and value types");
                   llvm::report_fatal_error("Pattern failed due to unsupported types");
                   // Alternatively, could yield sentinel values or handle differently.
              }
          } else {
              // Handle other input types if necessary (e.g., integer input)
              op.emitError("Unsupported input element type");
              llvm::report_fatal_error("Pattern failed due to unsupported input type");
          }


          // Calculate global index: chunkIdx * chunkSize + localIdx
          // linalg.index dimensions correspond to *loops*, not tensor dimensions.
          // Since iterTypes are [parallel, reduction], dim 0 = chunkIdx, dim 1 = localIdx
          auto chunkIdx = b.create<mlir::linalg::IndexOp>(bodyLoc, 0); // Parallel loop index
          auto localIdx = b.create<mlir::linalg::IndexOp>(bodyLoc, 1); // Reduction loop index
          auto baseIdx  = b.create<mlir::arith::MulIOp>(bodyLoc, chunkIdx, cChunkSz);
          auto globalIdx = b.create<mlir::arith::AddIOp>(bodyLoc, baseIdx, localIdx);

          // Cast global index (type index) to the *required* index type (idxElemTy)
          auto globalIdxCasted = b.create<mlir::arith::IndexCastOp>(bodyLoc, idxElemTy, globalIdx);

          // Comparison (use OGT for standard argmax, adapt if needed)
          auto cmp = b.create<mlir::arith::CmpFOp>(
              bodyLoc, mlir::arith::CmpFPredicate::OGT, inValProcessed, curVal);

          // Select new value and index
          auto newVal = b.create<mlir::arith::SelectOp>(bodyLoc, cmp, inValProcessed, curVal);
          auto newIdx = b.create<mlir::arith::SelectOp>(bodyLoc, cmp, globalIdxCasted, curIdx);

          b.create<mlir::linalg::YieldOp>(bodyLoc, mlir::ValueRange{newVal, newIdx});
        });


    // --- 7. Second‑stage reduction over the intermediate results ----------
    // Get the original output tensor types (scalar tensors)
    auto finalValTy = mlir::cast<mlir::RankedTensorType>(valInitOperand->get().getType());
    auto finalIdxTy = mlir::cast<mlir::RankedTensorType>(idxInitOperand->get().getType());


    // Create empty scalar tensors for the final output
    auto outValInit = rewriter.create<mlir::tensor::EmptyOp>(
        loc, finalValTy.getShape(), finalValTy.getElementType())
        .getResult();
     auto outIdxInit = rewriter.create<mlir::tensor::EmptyOp>(
        loc, finalIdxTy.getShape(), finalIdxTy.getElementType())
        .getResult();

    // Fill the scalar tensors with initial zero values
    auto valInitFilled =
        rewriter.create<mlir::linalg::FillOp>(loc, zeroVal, outValInit)
            .getResult(0);
    auto idxInitFilled =
        rewriter.create<mlir::linalg::FillOp>(loc, zeroIdx, outIdxInit)
            .getResult(0);


    // Input 1 (vals1): (d0) -> (d0) [nChunks] -> [nChunks]
    // Input 2 (idxs1): (d0) -> (d0) [nChunks] -> [nChunks]
    // Output 1 (val):  (d0) -> ()   [nChunks] -> []
    // Output 2 (idx):  (d0) -> ()   [nChunks] -> []
    auto mapInOut = mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, ctx);
    auto mapOutScalar = mlir::AffineMap::get(1, 0, {}, ctx); // Project reduction dim to scalar
    auto maps2 = llvm::SmallVector<mlir::AffineMap, 4>{mapInOut, mapInOut, mapOutScalar, mapOutScalar};

    auto iterTypes2 = llvm::SmallVector<mlir::utils::IteratorType>{
        mlir::utils::IteratorType::reduction // Reduce along the only dimension (nChunks)
    };

    auto secondStage = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/mlir::TypeRange{finalValTy, finalIdxTy},
        /*inputs=*/mlir::ValueRange{firstStage.getResult(0), firstStage.getResult(1)},
        /*outputs=*/mlir::ValueRange{valInitFilled, idxInitFilled},
        maps2, iterTypes2,
        [&](mlir::OpBuilder &b, mlir::Location bodyLoc, mlir::ValueRange args) {
          // args: input_val, input_idx, current_max_val, current_max_idx
          mlir::Value inVal  = args[0]; // Type valElemTy
          mlir::Value inIdx  = args[1]; // Type idxElemTy
          mlir::Value curVal = args[2]; // Type valElemTy
          mlir::Value curIdx = args[3]; // Type idxElemTy

          // Comparison
          auto cmp = b.create<mlir::arith::CmpFOp>(
              bodyLoc, mlir::arith::CmpFPredicate::OGT, inVal, curVal);

          // Select new value and index
          auto newVal = b.create<mlir::arith::SelectOp>(bodyLoc, cmp, inVal, curVal);
          auto newIdx = b.create<mlir::arith::SelectOp>(bodyLoc, cmp, inIdx, curIdx);

          b.create<mlir::linalg::YieldOp>(bodyLoc, mlir::ValueRange{newVal, newIdx});
        });


    // --- 8. Replace original op -------------------------------------------
    rewriter.replaceOp(op, secondStage->getResults());
    LLVM_DEBUG(llvm::dbgs() << "Successfully applied ArgMaxSplitReductionPattern\n");
    return mlir::success();
  }

private:
  int64_t chunkSize;
};


namespace {
struct SplitReductionPass final
    : public impl::SplitReductionPassBase<SplitReductionPass> {
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ArgMaxSplitReductionPattern>(context);
    if (mlir::failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns)))) {
        llvm::errs() << "failed patterns\n";
    }
  /*  if (splitReductionRatio.getValue() <= 1 &&*/
  /*      topkSplitReductionRatio.empty()) {*/
  /*    return;*/
  /*  }*/
  /**/
  /*  auto matmulSplitReductionControlFn =*/
  /*      [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {*/
  /*    // For matmul make the new parallel dimension first so that it looks*/
  /*    // like a batch_matmul and can follow the same codegen.*/
  ///*    return {int64_t(splitReductionRatio), 0, /*innerParallel=*/false};*/
  /*  };*/
  /**/
  /*  SmallVector<linalg::MatmulOp> matmulCandidates;*/
  /*  IRRewriter rewriter(context);*/
  /*  funcOp->walk([&](linalg::MatmulOp op) { matmulCandidates.push_back(op); });*/
  /*  for (auto op : matmulCandidates) {*/
  /*    (void)splitReductionOnMatmul(rewriter, op, matmulSplitReductionControlFn);*/
  /*  }*/
  /**/
  /*  IREE::LinalgExt::TopkSplitReductionControlFn topkSplitReductionControlFn =*/
  /*      [&](int64_t splitReductionDepth) -> int64_t {*/
  /*    SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),*/
  /*                                         topkSplitReductionRatio.end());*/
  /*    if (splitReductionDepth >= reductionRatios.size()) {*/
  /*      return -1;*/
  /*    } else {*/
  /*      return reductionRatios[splitReductionDepth];*/
  /*    }*/
  /*  };*/
  /**/
  /*  SmallVector<IREE::LinalgExt::TopkOp> topkCandidates;*/
  /*  funcOp->walk(*/
  /*      [&](IREE::LinalgExt::TopkOp op) { topkCandidates.push_back(op); });*/
  /*  for (auto op : topkCandidates) {*/
  /*    (void)splitReduction(rewriter, op, topkSplitReductionControlFn);*/
  /*  }*/
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
