// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- VideoCoreConfig.cpp - VideoCore CodeGen Configurations -------------===//
//
// This file contains CodeGen configurations for Broadcom VideoCore GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstdint>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-spirv-videocore-config"

using CodeGenPipeline =
    mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline;
namespace mlir::iree_compiler::detail {

static int64_t findLargestPowerOfTwoMultiple(int64_t dim, int64_t startValue) {
  if (dim < 0) {
    return 1;
  }
  while (dim % startValue != 0) {
    startValue >>= 1;
  }
  return std::max(startValue, 1ll);
}

LogicalResult
setMatmulOpVideoCoreConfig(IREE::GPU::TargetAttr target, linalg::LinalgOp op,
                           std::array<int64_t, 2> bestWorkgroupSizeXY,
                           std::array<int64_t, 3> bestThreadTileSizeMNK) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as matmul...\n");
  OpOperand *lhs = op.getDpsInputOperand(0);
  OpOperand *rhs = op.getDpsInputOperand(1);

  auto lhsType = llvm::cast<ShapedType>(lhs->get().getType());
  auto rhsType = llvm::cast<ShapedType>(rhs->get().getType());
  auto elementBits =
      static_cast<int>(IREE::Util::getTypeBitWidth(lhsType.getElementType()));
  if (!llvm::is_contained({8, 16, 32}, elementBits))
    return failure();

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic))
    return failure();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic))
    return failure();

  assert(llvm::is_contained({2u, 3u}, op.getNumParallelLoops()));

  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0)
    return failure();
  const bool isBM = bIndex >= 0;

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  const unsigned numLoops = loopRanges.size();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimN = loopRanges[nIndex];
  const int64_t dimK = loopRanges[kIndex];

  // Print out the input settings in debug mode
  int64_t bestX = bestWorkgroupSizeXY[0], bestY = bestWorkgroupSizeXY[1];
  LLVM_DEBUG({
    llvm::dbgs() << "best thread tile size (M, N, K) = ("
                 << bestThreadTileSizeMNK[0] << ", " << bestThreadTileSizeMNK[1]
                 << ", " << bestThreadTileSizeMNK[2] << ")\n";
    llvm::dbgs() << "best workgroup size (X, Y) = (" << bestX << ", " << bestY
                 << ")\n";
  });

  // The best workgroup size is around 256 in total. WG_Y <= 16 and WG_X*WG_Y ==
  // 256. Meaning the workgroup size should be balanced according to the output
  // shape. Through initial testing the best workgroup size per dimension is at
  // maximum 16. So we find the largest power of two that is smaller than the
  // largest dimension for the Y group dimension.
  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  workgroupSize[1] = std::min(dimN >> 1, 16ll);
  workgroupSize[0] =
      std::min(std::max((bestX * bestY) / workgroupSize[1], 1ll), dimM);

  SmallVector<int64_t> workgroupTileSizes(numLoops, 0);
  if (isBM)
    workgroupTileSizes[bIndex] = 1;
  workgroupTileSizes[mIndex] =
      findLargestPowerOfTwoMultiple(dimM, workgroupSize[0] * 4);
  workgroupTileSizes[nIndex] =
      findLargestPowerOfTwoMultiple(dimN, 1024 / workgroupTileSizes[mIndex]);

  SmallVector<int64_t> threadTileSizes(numLoops, 0);
  if (isBM) {
    threadTileSizes[bIndex] = workgroupTileSizes[bIndex] / workgroupSize[2];
  }
  threadTileSizes[mIndex] = std::max(workgroupTileSizes[mIndex] / 16, 1ll);
  threadTileSizes[nIndex] = std::max(workgroupTileSizes[nIndex] / 16, 1ll);

  SmallVector<int64_t> reductionTileSizes(numLoops, 0);
  int64_t maxVectorization = 32;
  reductionTileSizes[kIndex] =
      std::min(findLargestPowerOfTwoMultiple(maxVectorization, dimK), 4ll);

  workgroupTileSizes.resize(lastParallelDim + 1);
  threadTileSizes.resize(lastParallelDim + 1);

  TileSizesListType tileSizes;
  llvm::append_values(tileSizes, workgroupTileSizes, threadTileSizes,
                      reductionTileSizes);

  LLVM_DEBUG({
    llvm::dbgs() << "workgroup size (X, Y, X) = (" << workgroupSize[0] << ", "
                 << workgroupSize[1] << ", " << workgroupSize[2] << ")\n";
    llvm::dbgs() << "workgroup tiling (M, N) = (" << workgroupTileSizes[mIndex]
                 << ", " << workgroupTileSizes[nIndex] << ")\n";
    llvm::dbgs() << "thread tiling (M, N) = (" << threadTileSizes[mIndex]
                 << ", " << threadTileSizes[nIndex] << ")\n";
    llvm::dbgs() << "reduction tiling (M, N, k) = ("
                 << reductionTileSizes[mIndex] << ", "
                 << reductionTileSizes[nIndex] << ','
                 << reductionTileSizes[kIndex] << ")\n";
  });

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
      CodeGenPipeline::SPIRVBaseVectorize, workgroupSize);
}

static LogicalResult setVideoCoreMatmulConfig(linalg::LinalgOp op,
                                              IREE::GPU::TargetAttr target) {
  auto inputType =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  // Restrict the tilings to just float32 as we have not investigated the
  // optimal tiling for f16 or other types yet.
  if (!inputType.getElementType().isF32()) {
    return failure();
  }
  const std::array<int64_t, 2> workgroupXY = {16, 16};
  const std::array<int64_t, 3> threadMNK = {4, 4, 4};
  return setMatmulOpVideoCoreConfig(target, op, workgroupXY, threadMNK);
}

static LogicalResult setConvOpForVideoCoreConfig(linalg::LinalgOp op,
                                                 int64_t threadsPerWorkGroup,
                                                 int64_t optimalThreadTiling) {
  auto convDimsOrFailure = linalg::inferConvolutionDims(op);
  if (failed(convDimsOrFailure))
    return failure();
  auto inputType =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  // Restrict the tilings to just float32 as we have not investigated the
  // optimal tiling for f16 or other types yet.
  if (!inputType.getElementType().isF32()) {
    return failure();
  }
  const mlir::linalg::ConvolutionDimensions &convDims = *convDimsOrFailure;

  LLVM_DEBUG({
    llvm::dbgs() << "conv: " << op;
    llvm::dbgs() << "\nconv batch dim: ";
    llvm::interleaveComma(convDims.batch, llvm::dbgs());
    llvm::dbgs() << "\nconv output window dims: ";
    llvm::interleaveComma(convDims.outputImage, llvm::dbgs());
    llvm::dbgs() << "\nconv output channel dim: ";
    llvm::interleaveComma(convDims.outputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv filter window dims: ";
    llvm::interleaveComma(convDims.filterLoop, llvm::dbgs());
    llvm::dbgs() << "\nconv input channel dims: ";
    llvm::interleaveComma(convDims.inputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv depth multiplier: ";
    llvm::interleaveComma(convDims.depth, llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  const int ohIndex = convDims.outputImage.front();
  const int owIndex = convDims.outputImage.back();
  const int64_t oh = loopRanges[ohIndex];
  const int64_t ow = loopRanges[owIndex];

  int ocIndex;
  if (!convDims.outputChannel.empty()) {
    assert(convDims.outputChannel.size() == 1);
    ocIndex = convDims.outputChannel.front();
  } else if (!convDims.depth.empty()) {
    // For depthwise convolution ops with multipler 1, we have the same
    // input/filter/output channel size, which is being categorized as the
    // multipler.
    assert(convDims.depth.size() == 1);
    ocIndex = convDims.depth.front();
  } else {
    // For pooling ops, the input/output channel size will be categorized
    // as the additional batch dimension.
    assert(convDims.batch.size() == 2);
    ocIndex = convDims.batch.back();
  }

  Type outputType = op.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = llvm::cast<ShapedType>(outputType).getShape();
  if ((convDims.inputChannel.empty() ||
       ShapedType::isDynamic(convDims.inputChannel.front())) ||
      llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return failure();
  }

  // Output tilings for the SPIRV Vectorize Pipeline
  TileSizesListType tileSizes;
  const int64_t outputChannelSize = loopRanges[ocIndex];

  const bool isNCHW = ocIndex < ohIndex;
  if (!isNCHW) {
    // TODO(ROO-89): implement the NHWC case
    return failure();
  }

  // Workgroup tiling is very simple for large OC tensors we just use the
  // multiple of threads over channel
  SmallVector<int64_t> workgroupTiling(4, 1); // (N, OC, OH, OW)
  workgroupTiling[3] = findLargestPowerOfTwoMultiple(ow, threadsPerWorkGroup);
  threadsPerWorkGroup = threadsPerWorkGroup - workgroupTiling[3];
  workgroupTiling[2] = findLargestPowerOfTwoMultiple(oh, threadsPerWorkGroup);
  threadsPerWorkGroup = threadsPerWorkGroup - workgroupTiling[2];
  workgroupTiling[1] =
      findLargestPowerOfTwoMultiple(outputChannelSize, threadsPerWorkGroup);
  tileSizes.push_back(workgroupTiling);

  // Remember OC->Z, OW->Y and OH->X
  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  workgroupSize[0] = std::max(workgroupTiling[3] / 4, 1ll);
  workgroupSize[1] = workgroupTiling[2];
  workgroupSize[2] = std::max(workgroupTiling[1] / 4, 1ll);

  // Thread tiling seems to be most optimal when the last dim is tiled.
  SmallVector<int64_t> threadTiling = {1, 1, 1,
                                       optimalThreadTiling}; // (N, OC, OH, OW)
  tileSizes.push_back(threadTiling);

  // optimal tiling is on IC, F, F
  SmallVector<int64_t> reductionTiling(loopRanges.size(), 0);
  reductionTiling[convDims.inputChannel.front()] = 4;
  reductionTiling[convDims.filterLoop.front()] = 1;
  reductionTiling[convDims.filterLoop.back()] = 1;
  tileSizes.push_back(reductionTiling);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);

  auto pipeline = CodeGenPipeline::SPIRVBaseVectorize;
  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setVideoCoreCodeGenConfig(IREE::GPU::TargetAttr target,
                                        Operation *rootOp) {
  if (!isa<linalg::LinalgOp>(rootOp))
    return failure();

  auto linalgOp = cast<linalg::LinalgOp>(rootOp);
  if (isMatmulOrBatchMatmul(linalgOp) || isa<linalg::MatmulOp>(linalgOp) ||
      isa<linalg::BatchMatmulOp>(linalgOp)) {
    return setVideoCoreMatmulConfig(linalgOp, target);
  }

  // The IREE heuristic got us quite good performance around 60s for ResNet18
  // using the below settings. if (auto convOp =
  // dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
  //   return setConvOpConfig(cast<linalg::LinalgOp>(rootOp), 256, 4));
  // }

  // With this heuristic we get around 2.4s with ResNet18 on the RPI5 GPU
  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    return setConvOpForVideoCoreConfig(cast<linalg::LinalgOp>(rootOp), 256, 4);
  }

  return failure();
}

} // namespace mlir::iree_compiler::detail
