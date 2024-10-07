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
  return setMatmulOpConfig(target, op, workgroupXY, threadMNK);
}

static int64_t getWorkgroupTiling(int64_t &remainingThreads,
                                  int64_t dimensionSize) {
  // Find the largest power of two value that can evenly divide the dimension
  // and take that away from the pool of available threads.
  auto result = dimensionSize & (~(dimensionSize - 1));
  remainingThreads = std::max(result / remainingThreads, 1ll);
  return result;
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
  workgroupTiling[3] = getWorkgroupTiling(threadsPerWorkGroup, ow);
  workgroupTiling[2] = getWorkgroupTiling(threadsPerWorkGroup, oh);
  workgroupTiling[1] =
      getWorkgroupTiling(threadsPerWorkGroup, outputChannelSize);
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

  // With this heuristic we get around 3.2s with ResNet18 on the RPI5 GPU
  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    return setConvOpForVideoCoreConfig(cast<linalg::LinalgOp>(rootOp), 256, 4);
  }

  return failure();
}

} // namespace mlir::iree_compiler::detail
