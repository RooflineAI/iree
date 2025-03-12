// Copyright 2021 The IREE Authors
// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include <cstdint>
#include <numeric>
#include <optional>

#include "compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "roof-cuda-gpu-kernel-config"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::iree_compiler;

namespace cellar::target::cuda {

namespace {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

// Threshold used to determine whether a matmul dimension is 'very skinny'.
constexpr int64_t kVerySkinnyDimThreshold = 4;

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
  int64_t pipelineDepth;
};

// Simt codegen does not do software pipelining.
constexpr unsigned softwarePipelineDepthSimt = 0;

} // namespace

//====---------------------------------------------------------------------===//
// Matmul Configuration Helpers
//====---------------------------------------------------------------------===//

/// Return the best combination of tile size and wg size. It will then used to
/// pick the best size aligned with the shape dimension.
static SmallVector<TileWorkgroupSizePair>
getMatmulConfig(IREE::GPU::TargetAttr target) {
  SmallVector<TileWorkgroupSizePair> tileSizes;
  // Pick tile size so that M*K and K*N dividible by wgSize * \*vecSize=*\4.
  // This way workgroup memory copy don't need to be masked. Once we support
  // masked load we can get performance out of more configuration.

  // Make use of the full subgroup when possible.
  if (target.getPreferredSubgroupSize() == 64) {
    tileSizes.push_back(TileWorkgroupSizePair({{64, 128, 64}, {64, 16, 1}, 1}));
  }

  llvm::append_values(tileSizes,
                      TileWorkgroupSizePair({{32, 128, 32}, {32, 8, 1}, 1}),
                      TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}, 1}),
                      TileWorkgroupSizePair({{16, 256, 32}, {64, 2, 1}, 1}),
                      TileWorkgroupSizePair({{8, 32, 32}, {8, 8, 1}, 1}),

                      TileWorkgroupSizePair({{32, 128, 4}, {32, 8, 1}, 1}),
                      TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}, 1}),
                      TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}, 1}),
                      TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}, 1}));
  return tileSizes;
}

/// Return the best combination of tile size and wg size when using tensorcore
/// operations.
static void
getTensorCoreConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes,
                    Type elementType, int64_t M, int64_t N, int64_t K) {
  // Based on early analysis we found that 128x256x32_3 gives acceptable
  // performance across many of the large matrix sizes for f16 and fp32. This
  // needs to be refined into a better startegy based on empircal data but this
  // gives us a quick solution to achieve performance in the right order of
  // magnitude for large square like cases.
  int64_t parallelDim = M * N;
  static constexpr int64_t kLargDimThreashold = 1536;
  if (elementType.isF16()) {
    if (parallelDim >= kLargDimThreashold * kLargDimThreashold) {
      tileSizes.push_back(
          TileWorkgroupSizePair({{128, 256, 32}, {128, 2, 1}, 3}));
    }
    tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 32}, {64, 2, 1}, 4}));
  } else {
    if (parallelDim >= kLargDimThreashold * kLargDimThreashold) {
      llvm::append_values(
          tileSizes, TileWorkgroupSizePair({{128, 256, 16}, {128, 2, 1}, 4}),
          TileWorkgroupSizePair({{64, 128, 16}, {64, 2, 1}, 4}));
    }
    llvm::append_values(tileSizes,
                        TileWorkgroupSizePair({{32, 32, 16}, {64, 2, 1}, 4}),
                        TileWorkgroupSizePair({{16, 32, 16}, {64, 1, 1}, 4}),
                        TileWorkgroupSizePair({{32, 16, 16}, {32, 2, 1}, 4}),
                        TileWorkgroupSizePair({{16, 16, 16}, {32, 1, 1}, 4}));
  }
}

static bool supportsTensorCore(IREE::GPU::TargetAttr target,
                               linalg::LinalgOp op) {
  // Limit tensor core pipeline to matmul as not all combinations of transpose
  // are supported upstream.
  if (!target.supportsSyncMMAOps())
    return false;
  if (!(isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op))) {
    assert(linalg::isaContractionOpInterface(op));
    // If this is not a named op matmul check some properties to make sure that
    // we can map it to tensorcore ops. We should have only mulAdd in the region
    // and the output map should have no permutation and the last dimension
    // should be a reduce.
    Region &body = op->getRegion(0);
    Region::OpIterator it = body.op_begin();
    if (it == body.op_end() || !isa<arith::MulFOp>(*(it++)))
      return false;
    if (it == body.op_end() || !isa<arith::AddFOp>(*(it++)))
      return false;
    if (it == body.op_end() || !isa<linalg::YieldOp>(*(it++)))
      return false;
    AffineMap outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
    if (outputMap.getNumResults() != outputMap.getNumDims() - 1)
      return false;
    OpBuilder b(op);
    for (unsigned i = 0, e = outputMap.getNumResults(); i < e - 1; i++) {
      if (outputMap.getResult(i) != b.getAffineDimExpr(i))
        return false;
    }
  }
  return true;
}

/// Decides which tensorcore operations to use.
static CodeGenPipeline getTensorCorePipeline(Type elementType) {
  // Currently mma.sync is on by default for fp16 only.
  CodeGenPipeline codegenPipeline = CodeGenPipeline::LLVMGPUMatmulTensorCore;

  // For F16 and F32 use mmasync by default.
  if (elementType.isF16() || elementType.isF32()) {
    codegenPipeline = CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync;
  }

  return codegenPipeline;
}

//====---------------------------------------------------------------------===//
// Vector Distribution Contraction/Convolution Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult
setConvolutionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                       mlir::FunctionOpInterface entryPoint,
                                       linalg::LinalgOp op) {
  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
      mlir::linalg::inferConvolutionDims(op);
  if (failed(convolutionDims)) {
    return failure();
  }

  // This strategy turns non-strided/dilated convolution problems into matmul
  // problems by tiling certain dimensions to 1:
  //  - Batch dimensions (parallel shared by the image and output)
  //  - Filter dimensions (reduction on the filter, and convolved on the image)
  //  - All output image dimensions except the outermost one
  //
  // After this, the remaining non-unit dimensions are:
  //  - One output image dimension corresponding to the M dimension of a matmul.
  //  - The output channel dimension, corresponding to the N dimension.
  //  - The input channel dimension, corresponding to the K dimension.

  // TODO: Relax this condition to strictly alignment requirements.
  if (convolutionDims->outputChannel.size() < 1 ||
      convolutionDims->inputChannel.size() < 1 ||
      convolutionDims->filterLoop.size() < 1 ||
      convolutionDims->outputImage.size() < 1 ||
      convolutionDims->depth.size() != 0) {
    return failure();
  }

  auto isAllOnesList = [](ArrayRef<int64_t> list) {
    return llvm::all_of(list, [](int64_t i) { return i == 1; });
  };

  // TODO: Support non-unit strides/dilations.
  if (!isAllOnesList(convolutionDims->strides) ||
      !isAllOnesList(convolutionDims->dilations)) {
    return failure();
  }

  int64_t mDim = convolutionDims->outputImage.back();
  int64_t nDim = convolutionDims->outputChannel.back();
  // TODO: Support NCHW convolutions. This is just a matmul_transpose_a, however
  // the distribution patterns currently do not support that variant.
  if (mDim > nDim) {
    return failure();
  }
  int64_t kDim = convolutionDims->inputChannel.back();

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0] and
  // schedule->m/n/kTileSizes[0] need to use the full list of sizes instead of
  // just the first element.
  GPUMatmulShapeType problem{bounds[mDim], bounds[nDim], bounds[kDim],
                             lhsElemType,  rhsElemType,  initElemType};

  // Helper fn to store mma information.
  auto storeMmaInfo = [](IREE::GPU::MmaInterfaceAttr mma,
                         SmallVector<GPUMatmulShapeType> &intrinsics,
                         SmallVector<IREE::GPU::MmaInterfaceAttr> &mmaKinds) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
    mmaKinds.emplace_back(mma);
  };

  SmallVector<GPUMatmulShapeType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  SmallVector<IREE::GPU::MmaInterfaceAttr> mmaKinds;
  MLIRContext *context = op.getContext();
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    storeMmaInfo(mma, intrinsics, mmaKinds);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics, mmaKinds);
    }
  }

  if (intrinsics.empty())
    return failure();

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  GPUMMAHeuristicSeeds seeds{/*bestSubgroupCountPerWorkgroup=*/4,
                             /*bestMNTileCountPerSubgroup=*/8,
                             /*bestKTileCountPerSubgroup=*/2};

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  // First try to find a schedule with an exactly matching intrinsic.
  FailureOr<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize);
  if (failed(schedule)) {
    // Then try again by allowing upcasting accumulator.
    schedule = deduceMMASchedule(
        problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
        /*transposedLhs*/ false, /*transposedRhs*/ false,
        /*canUpcastAcc=*/true);
  }
  if (failed(schedule)) {
    return failure();
  }

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : convolutionDims->batch) {
    workgroupTileSizes[batch] = 1;
  }
  // Tile all output image dimensions with unit size except the last one.
  for (int64_t oi : llvm::drop_end(convolutionDims->outputImage)) {
    workgroupTileSizes[oi] = 1;
  }
  for (int64_t oc : llvm::drop_end(convolutionDims->outputChannel)) {
    workgroupTileSizes[oc] = 1;
  }
  for (int64_t ic : llvm::drop_end(convolutionDims->inputChannel)) {
    reductionTileSizes[ic] = 1;
  }
  // Compute the M/N dimension tile size by multiply subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * schedule->nSize;

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSize;

  // Tile all filter loop dimensions to 1.
  for (int64_t filterDim : convolutionDims->filterLoop) {
    reductionTileSizes[filterDim] = 1;
  }

  Builder b(context);
  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  IREE::GPU::appendPromotedOperandsList(context, attrs, {0, 1});
  IREE::GPU::setMmaKind(context, attrs, mmaKinds[schedule->index]);
  IREE::GPU::setSubgroupMCount(context, attrs, schedule->mSubgroupCounts[0]);
  IREE::GPU::setSubgroupNCount(context, attrs, schedule->nSubgroupCounts[0]);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;

  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, CodeGenPipeline::LLVMGPUVectorDistribute,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

[[maybe_unused]] static void
debugPrintContractionInfo(StringRef label, unsigned numLoops,
                          linalg::ContractionDimensions contractionDims,
                          ArrayRef<int64_t> sizes) {
  ArrayRef<unsigned> dimVals[] = {contractionDims.batch, contractionDims.m,
                                  contractionDims.n, contractionDims.k};
  std::string dimSymbols(numLoops, '*');
  for (auto [idx, val] : llvm::enumerate(dimSymbols)) {
    for (auto [letter, dim] : llvm::zip_equal(StringRef("bmnk"), dimVals))
      if (llvm::is_contained(dim, idx))
        val = letter;
  }
  DBGS() << "Contraction dims: [";
  llvm::interleaveComma(dimSymbols, llvm::dbgs());
  llvm::dbgs() << "]\n";

  DBGS() << label << ": [";
  llvm::interleaveComma(sizes, llvm::dbgs());
  llvm::dbgs() << "]\n";
}

static LogicalResult
setMatmulVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                  mlir::FunctionOpInterface entryPoint,
                                  linalg::LinalgOp op) {
  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);
  if (failed(contractionDims)) {
    assert(IREE::LinalgExt::isaHorizontallyFusedContraction(op) &&
           "expected horizontally fused contraction op");
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInputOperand(1)));
    indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInitOperand(0)));
    contractionDims = mlir::linalg::inferContractionDims(indexingMaps);
  }
  assert(succeeded(contractionDims) && "Could not infer contraction dims");

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return failure();
  }

  LLVM_DEBUG(debugPrintContractionInfo("Problem size", op.getNumLoops(),
                                       *contractionDims, bounds));

  // For now we are not being smart and trying to reshape dimensions to allow
  // for better usage of intrinsics, and instead are tiling all dimensions
  // except the inner most m, n, and k dimensions to 1.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // Dynamic dims are expected to be taken care of earlier in the pipeline.
  if (ShapedType::isDynamic(bounds[mDim]) ||
      ShapedType::isDynamic(bounds[nDim]) ||
      ShapedType::isDynamic(bounds[kDim])) {
    return failure();
  }

  // Bail out on matvec-like cases.
  if (bounds[mDim] == 1 || bounds[nDim] == 1) {
    return failure();
  }

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  if (auto lhsOp = lhs.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(lhsOp))
      lhsElemType = getElementTypeOrSelf(lhsOp.getDpsInputs()[0]);
  }
  if (auto rhsOp = rhs.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(rhsOp))
      rhsElemType = getElementTypeOrSelf(rhsOp.getDpsInputs()[0]);
  }

  SmallVector<int64_t> batchDims;
  for (int64_t batchDim : contractionDims->batch) {
    if (!ShapedType::isDynamic(bounds[batchDim])) {
      batchDims.push_back(batchDim);
    }
  }
  auto getDimBounds = [&](SmallVector<int64_t> dims) -> SmallVector<int64_t> {
    return llvm::map_to_vector(dims, [&](int64_t dim) { return bounds[dim]; });
  };

  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0] and
  // schedule->m/n/kTileSizes[0] need to use the full list of sizes instead of
  // just the first element.
  GPUMatmulShapeType problem{
      {bounds[mDim]}, {bounds[nDim]}, {bounds[kDim]}, getDimBounds(batchDims),
      lhsElemType,    rhsElemType,    initElemType};

  // Helper fn to store mma information.
  auto storeMmaInfo = [](IREE::GPU::MmaInterfaceAttr mma,
                         SmallVector<GPUMatmulShapeType> &intrinsics,
                         SmallVector<IREE::GPU::MmaInterfaceAttr> &mmaKinds) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
    mmaKinds.emplace_back(mma);
  };

  SmallVector<GPUMatmulShapeType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  SmallVector<IREE::GPU::MmaInterfaceAttr> mmaKinds;
  MLIRContext *context = op.getContext();
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    storeMmaInfo(mma, intrinsics, mmaKinds);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics, mmaKinds);
    }
  }

  if (intrinsics.empty())
    return failure();

  GPUMMAHeuristicSeeds seeds;

  seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
            /*bestMNTileCountPerSubgroup=*/8,
            /*bestKTileCountPerSubgroup=*/4};
  // Scale the seed by number of contractions of horizontally fused case.
  seeds.bestMNTileCountPerSubgroup /= op.getNumDpsInputs() - 1;

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  LDBG("Matmul Vector Distribution Config");

  auto pipeline = CodeGenPipeline::LLVMGPUVectorDistribute;

  // Infer if lhs or rhs is transposed to help generate better schedule.
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  bool transposedLhs =
      kDim !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDim !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize);
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule =
        deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                          targetSubgroupSize, transposedLhs, transposedRhs,
                          /*canUpcastAcc=*/true);
  }

  // Only batch_matmul is supported in the LLVMGPUPadAndVectorDistribute
  // pipeline.
  // TODO(hanchung): Support cases that there are fused producers.
  if (!schedule && !contractionDims->batch.empty() && !hasFusedLeadingOp(op)) {
    LDBG("Matmul Pad and Vector Distribute");
    pipeline = CodeGenPipeline::LLVMGPUPadAndVectorDistribute;
    bool mustBeAligned = false;
    schedule =
        deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                          targetSubgroupSize, transposedLhs, transposedRhs,
                          /*canUpcastAcc=*/false, mustBeAligned);
    if (!schedule) {
      // Then try again by allowing upcasting accumulator.
      schedule =
          deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                            targetSubgroupSize, transposedLhs, transposedRhs,
                            /*canUpcastAcc=*/true, mustBeAligned);
    }
  }
  if (!schedule) {
    LDBG("Failed to deduce MMA schedule");
    return failure();
  }

  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: " << schedule);

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionDims->batch) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims->k)) {
    reductionTileSizes[k] = 1;
  }

  // Compute the M/N dimension tile size by multiply subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * schedule->nSize;

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSize;

  LLVM_DEBUG(debugPrintContractionInfo("Workgroup tile sizes", op.getNumLoops(),
                                       *contractionDims, workgroupTileSizes));
  LLVM_DEBUG(debugPrintContractionInfo("Reduction tile sizes", op.getNumLoops(),
                                       *contractionDims, reductionTileSizes));

  Builder b(context);
  SmallVector<NamedAttribute, 2> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  auto promotedOperands =
      llvm::to_vector(llvm::seq<int64_t>(op.getNumDpsInputs()));
  IREE::GPU::appendPromotedOperandsList(context, attrs, promotedOperands);
  IREE::GPU::setMmaKind(context, attrs, mmaKinds[schedule->index]);
  IREE::GPU::setSubgroupMCount(context, attrs, schedule->mSubgroupCounts[0]);
  IREE::GPU::setSubgroupNCount(context, attrs, schedule->nSubgroupCounts[0]);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  SmallVector<NamedAttribute, 1> pipelineAttrs;

  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig, pipeline, workgroupSize,
      targetSubgroupSize, pipelineConfig);
}

static LogicalResult
setVectorDistributionConfig(IREE::GPU::TargetAttr target,
                            mlir::FunctionOpInterface entryPoint,
                            Operation *computeOp) {
  LDBG("VectorDistribution: finding a suitable config...");

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (linalg::isaContractionOpInterface(linalgOp) ||
        IREE::LinalgExt::isaHorizontallyFusedContraction(linalgOp)) {
      LDBG("VectorDistribution: trying to find a suitable contraction config");
      return setMatmulVectorDistributionConfig(target, entryPoint, linalgOp);
    }
    if (linalg::isaConvolutionOpInterface(linalgOp)) {
      LDBG("VectorDistribution: trying to find a suitable convolution config");
      return setConvolutionVectorDistributionConfig(target, entryPoint,
                                                    linalgOp);
    }
  }

  LDBG("VectorDistribution: failed to find a suitable config");
  return failure();
}

//====---------------------------------------------------------------------===//
// Contraction Pipeline Configuration
//====---------------------------------------------------------------------===//

// Checks whether the giving tiling will fit within the GPU shared memory.
static bool doesMatMulTileFitInSharedMem(const TileWorkgroupSizePair &config,
                                         Type element,
                                         IREE::GPU::TargetAttr target) {
  unsigned int bytesPerElement = element.getIntOrFloatBitWidth() / 8;
  // Given a potential tiling we can figure out the amount of memory each matrix
  // will take up in shared memory for a mat mul op, i.e. C = A x B
  unsigned int calculatedMem =
      // Find Matrix A size: MxK
      bytesPerElement * config.tileSize[0] * config.tileSize[2] +
      // Find Matrix B size: KxN
      bytesPerElement * config.tileSize[1] * config.tileSize[2] +
      // Find Matrix C size: MxN
      bytesPerElement * config.tileSize[0] * config.tileSize[1];
  return calculatedMem < target.getWgp().getMaxWorkgroupMemoryBytes();
}

static LogicalResult setContractConfig(IREE::GPU::TargetAttr target,
                                       mlir::FunctionOpInterface entryPoint,
                                       linalg::LinalgOp op) {
  if (!linalg::isaContractionOpInterface(op) || op.getNumParallelLoops() < 2) {
    return failure();
  }

  // Also exclude the case of matvec, which has only one non-unit parallel dim.
  // They should go down different pipelines.
  // Currently dynamic dimensions are tiled with size=1 in codegen.
  int staticNonUnitParallelDimCount = 0;
  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);
  assert(succeeded(contractionDims) && "Could not infer contraction dims");
  for (auto mDim : contractionDims->m) {
    staticNonUnitParallelDimCount +=
        bounds[mDim] != 1 && !ShapedType::isDynamic(bounds[mDim]);
  }
  for (auto nDim : contractionDims->n) {
    staticNonUnitParallelDimCount +=
        bounds[nDim] != 1 && !ShapedType::isDynamic(bounds[nDim]);
  }
  if (staticNonUnitParallelDimCount <= 1)
    return failure();

  // Don't consider operations that don't have a broadcast, those should go
  // through reductions.
  if (llvm::any_of(op.getIndexingMapsArray(),
                   [](AffineMap m) { return m.isPermutation(); })) {
    return failure();
  }

  // Send very skinny, {2-4}xNxK and Mx{2-4}xK, matmuls to the vector reduction
  // pipeline, similar to matvec. Note: Because of reassociation in the vector
  // reduction pipeline, this may lead to precission loss. If this ever becomes
  // an issue, we can hide this behind a flag.
  if (llvm::all_equal({contractionDims->m.size(), contractionDims->n.size(),
                       contractionDims->k.size(), size_t{1}}) &&
      contractionDims->batch.empty()) {
    int64_t mSize = bounds[contractionDims->m.front()];
    int64_t nSize = bounds[contractionDims->n.front()];
    int64_t preferredSubgroupSize = target.getPreferredSubgroupSize();
    if ((mSize <= kVerySkinnyDimThreshold &&
         (nSize > preferredSubgroupSize || ShapedType::isDynamic(nSize))) ||
        (nSize <= kVerySkinnyDimThreshold &&
         (mSize > preferredSubgroupSize || ShapedType::isDynamic(mSize)))) {
      return failure();
    }
  }

  // TODO: Properly rematerialize leading elementwise with shared memory
  // promotion.
  if (hasFusedLeadingOp(op)) {
    return failure();
  }

  auto setMatmulConfig = [&entryPoint, &op](int64_t tileX, int64_t tileY,
                                            int64_t tileK,
                                            ArrayRef<int64_t> workgroupSize,
                                            ArrayRef<int32_t> subgroupSizes,
                                            unsigned softwarePipelineDepth,
                                            CodeGenPipeline pipeline) {
    TileSizesListType tileSizes;
    unsigned numParallelLoops = op.getNumParallelLoops();
    unsigned numReductionLoops = op.getNumReductionLoops();
    SmallVector<int64_t> workgroupTileSizes(
        numParallelLoops + numReductionLoops, 1);
    workgroupTileSizes[numParallelLoops - 2] = tileX;
    workgroupTileSizes[numParallelLoops - 1] = tileY;

    SmallVector<unsigned> partitionedLoops =
        cast<PartitionableLoopsInterface>(op.getOperation())
            .getPartitionableLoops(/*maxNumPartitionedLoops=*/std::nullopt);
    llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
    partitionedLoopsSet.insert(partitionedLoops.begin(),
                               partitionedLoops.end());
    for (auto loopID : llvm::seq<unsigned>(0, numParallelLoops)) {
      if (!partitionedLoopsSet.count(loopID)) {
        workgroupTileSizes[loopID] = 0;
      }
    }

    std::optional<int64_t> subgroupSize = std::nullopt;
    if (!subgroupSizes.empty())
      subgroupSize = subgroupSizes.front();

    // For the LLVMGPUTileAndFuse pipeline, we need to split tile sizes
    // for workgroup, thread, and reduction.
    if (pipeline == CodeGenPipeline::LLVMGPUTileAndFuse) {

      auto context = op.getContext();
      Builder b(context);

      SmallVector<int64_t> threadTileSizes(numParallelLoops + numReductionLoops,
                                           0);
      std::fill(threadTileSizes.begin(),
                threadTileSizes.begin() + numParallelLoops, 1);

      threadTileSizes[numParallelLoops - 2] =
          (tileX / workgroupSize[0]) < 1 ? 1 : (tileX / workgroupSize[0]);
      threadTileSizes[numParallelLoops - 1] =
          (tileY / workgroupSize[1]) < 1 ? 1 : (tileY / workgroupSize[1]);

      SmallVector<int64_t> reductionTileSizes(
          numParallelLoops + numReductionLoops, 0);
      reductionTileSizes[numParallelLoops + numReductionLoops - 1] = tileK;

      SmallVector<NamedAttribute, 3> attrs = {
          NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
          NamedAttribute("thread", b.getI64ArrayAttr(threadTileSizes)),
          NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes))};

      auto configDict = b.getDictionaryAttr(attrs);
      auto loweringConfig =
          IREE::GPU::LoweringConfigAttr::get(context, configDict);
      SmallVector<NamedAttribute, 1> pipelineAttrs;
      auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
          context, /*prefetchSharedMemory=*/false,
          /*no_reduce_shared_memory_bank_conflicts=*/true,
          /*use_igemm_convolution=*/false,
          /*reorder_workgroups_strategy=*/std::nullopt);
      pipelineAttrs.emplace_back(
          b.getStringAttr(IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
          pipelineOptions);
      auto pipelineConfig = b.getDictionaryAttr(pipelineAttrs);

      return setOpConfigAndEntryPointFnTranslation(
          entryPoint, op, loweringConfig, pipeline, workgroupSize, subgroupSize,
          pipelineConfig);
    }

    // Other pipeline (MatmulTensorCore) expect the reduction tile size to be in
    // the same list.
    workgroupTileSizes[numParallelLoops + numReductionLoops - 1] = tileK;
    tileSizes.emplace_back(std::move(workgroupTileSizes));

    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, pipeline, workgroupSize, subgroupSize,
        getSoftwarePipeliningAttrDict(op->getContext(), softwarePipelineDepth,
                                      /*softwarePipelineStoreStage=*/1));
  };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType())
          .getShape();
  auto rhsShape =
      llvm::cast<ShapedType>(op.getDpsInputOperand(1)->get().getType())
          .getShape();
  int64_t sizeM = ShapedType::kDynamic;
  int64_t sizeN = ShapedType::kDynamic;
  int64_t sizeK = ShapedType::kDynamic;
  auto outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = !ShapedType::isDynamic(sizeM) &&
                      !ShapedType::isDynamic(sizeN) &&
                      !ShapedType::isDynamic(sizeK);
  if (isStaticSize) {
    /// Try tensorcore config first.
    if (supportsTensorCore(target, op)) {
      SmallVector<TileWorkgroupSizePair> TCtileSizeConfig;
      Type elementType =
          cast<ShapedType>(op.getDpsInputOperand(0)->get().getType())
              .getElementType();

      getTensorCoreConfig(TCtileSizeConfig, elementType, sizeM, sizeN, sizeK);
      // Pick the best configuration where the original shape is aligned on the
      // tile size.
      for (TileWorkgroupSizePair &config : TCtileSizeConfig) {
        if (sizeK % config.tileSize[2] == 0 &&
            sizeN % config.tileSize[1] == 0 &&
            sizeM % config.tileSize[0] == 0 &&
            doesMatMulTileFitInSharedMem(config, elementType, target)) {
          CodeGenPipeline codegenPipeline = getTensorCorePipeline(elementType);
          return setMatmulConfig(
              config.tileSize[0], config.tileSize[1], config.tileSize[2],
              config.workgroupSize,
              target.getWgp().getSubgroupSizeChoices().asArrayRef(),
              sizeK == config.tileSize[2] ? 1 : config.pipelineDepth,
              codegenPipeline);
        }
      }
    }
    // Special case for very small matrices.
    if (sizeM * sizeN <= target.getPreferredSubgroupSize()) {
      return setMatmulConfig(
          sizeN, sizeM, 4, {sizeM, sizeN, 1},
          target.getWgp().getSubgroupSizeChoices().asArrayRef(),
          softwarePipelineDepthSimt, CodeGenPipeline::LLVMGPUTileAndFuse);
    }

    // SIMT matmul case. Query the best configuration.
    SmallVector<TileWorkgroupSizePair> tileSizeConfig = getMatmulConfig(target);
    // Pick the best configuration where the original shape is aligned on the
    // tile size.
    for (TileWorkgroupSizePair &config : tileSizeConfig) {
      if (sizeN % config.tileSize[1] == 0 && sizeM % config.tileSize[0] == 0 &&
          sizeK % config.tileSize[2] == 0) {
        return setMatmulConfig(
            config.tileSize[0], config.tileSize[1], config.tileSize[2],
            config.workgroupSize,
            target.getWgp().getSubgroupSizeChoices().asArrayRef(),
            softwarePipelineDepthSimt, CodeGenPipeline::LLVMGPUTileAndFuse);
      }
    }
  }
  // If we haven't found any config, use the best tile size hoping that
  // the workgroup specialization handles the main tile path efficiently.
  SmallVector<TileWorkgroupSizePair> tileSizeConfig = getMatmulConfig(target);
  constexpr size_t configIndex = 0;
  const TileWorkgroupSizePair &config = tileSizeConfig[configIndex];
  const int64_t tileX = config.tileSize[0];
  const int64_t tileY = config.tileSize[1];
  int64_t tileK = config.tileSize[2];
  // Since specialization doesn't work for K loop and peeling is not enabled yet
  // we pick a tileK size that is aligned on the K size.
  if (ShapedType::isDynamic(sizeK))
    tileK = 1;
  while (sizeK % tileK != 0) {
    tileK >>= 1;
  }
  const std::array<int64_t, 3> workgroupSize{config.workgroupSize[0],
                                             config.workgroupSize[1],
                                             config.workgroupSize[2]};
  return setMatmulConfig(tileX, tileY, tileK, workgroupSize,
                         target.getWgp().getSubgroupSizeChoices().asArrayRef(),
                         softwarePipelineDepthSimt,
                         CodeGenPipeline::LLVMGPUTileAndFuse);
}

//====---------------------------------------------------------------------===//
// Default Pipeline Configuration
//====---------------------------------------------------------------------===//

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(IREE::GPU::TargetAttr target,
                                          mlir::FunctionOpInterface entryPoint,
                                          Operation *op) {
  CodeGenPipeline passPipeline = CodeGenPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops = interfaceOp.getPartitionableLoops(std::nullopt);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                                 passPipeline, {1, 1, 1});
  }

  const int preferredSubgroupSize = target.getPreferredSubgroupSize();
  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps.
  std::array<int64_t, 3> workgroupSize = {2 * preferredSubgroupSize, 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }
  int64_t skipInnerTiling = 0;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (auto [index, outputOperand] :
         llvm::enumerate(genericOp.getDpsInitsMutable())) {
      if (!genericOp.getMatchingIndexingMap(&outputOperand)
               .isProjectedPermutation()) {
        vectorSize = 1;
        break;
      }
      ArrayRef<int64_t> shape =
          llvm::cast<ShapedType>(outputOperand.get().getType()).getShape();
      if (llvm::any_of(shape, ShapedType::isDynamic)) {
        vectorSize = 1;
        break;
      }
      // Since we vectorize along the most inner dimension, make sure if can be
      // divided by number of threads * vectorSize.
      while (vectorSize > 1 &&
             shape.back() % (workgroupSize[0] * vectorSize) != 0) {
        vectorSize /= 2;
      }
      if (vectorSize == 1) // assume there is fastpath + slowpath
        vectorSize = 4;
      int64_t problemSize = std::accumulate(
          shape.begin(), shape.end(), 1,
          [](const int64_t &a, const int64_t &b) { return a * b; });
      if ((problemSize / (preferredSubgroupSize * vectorSize)) < 64) {
        vectorSize = 1;
        break;
      }
      // If the inner dimension is too small to have one element per thread
      // reduce the workgroup size try to distribute amongst more dimensions.
      if (shape.back() < vectorSize * workgroupSize[0]) {
        int64_t flatWG = workgroupSize[0];
        vectorSize = 1;
        int64_t id = 0;
        for (int64_t dim : llvm::reverse(shape)) {
          // Unit loops are already skipped.
          if (dim == 1)
            continue;
          if (dim < flatWG) {
            skipInnerTiling++;
            workgroupSize[id] = dim;
          } else {
            workgroupSize[id] = flatWG;
            break;
          }
          flatWG = flatWG / dim;
          id++;
          if (flatWG <= 1 || id >= workgroupSize.size())
            break;
        }
        break;
      }
    }
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Pick a vectorSize of 1 for op that we know won't get vectorized.
  // Also skip vectorization for linalg on memref (no result) as the pipeline
  // relies on tensor level tiling.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!linalgOp || op->getNumResults() != 1 ||
      llvm::any_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    vectorSize = 1;
  } else {
    passPipeline = CodeGenPipeline::LLVMGPUVectorize;
  }

  int64_t id = 0;
  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      if (skipInnerTiling > 0) {
        // For dimensions that don't need to be distributed across blocks skip
        // tiling by setting tile size to 0.
        workgroupTileSizes[depth - 1] = 0;
        skipInnerTiling--;
        id++;
        if (id >= workgroupSize.size())
          break;
        continue;
      }
      workgroupTileSizes[depth - 1] = workgroupSize[id] * vectorSize;
      break;
    }
  }

  if (linalgOp) {
    // Tile reduction dimension to 4 to allow doing load4 if the reduction size
    // is the most inner dimension.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 4);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                               passPipeline, workgroupSize,
                                               preferredSubgroupSize);
}

//====---------------------------------------------------------------------===//
// Pipeline Configuration
//====---------------------------------------------------------------------===//

static LogicalResult setRootConfig(IREE::GPU::TargetAttr target,
                                   mlir::FunctionOpInterface entryPointFn,
                                   Operation *computeOp) {
  LLVM_DEBUG({
    DBGS() << "Selecting root config for: ";
    computeOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  if (succeeded(IREE::GPU::setTileAndFuseLoweringConfig(target, entryPointFn,
    computeOp))) {
    LDBG("Tile and fuse default config");
    return success();
  }
  if (succeeded(setVectorDistributionConfig(target, entryPointFn, computeOp))) {
    LDBG("VectorizeDistribute Config");
    return success();
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (succeeded(setContractConfig(target, entryPointFn, linalgOp))) {
      LDBG("Contract Config");
      return success();
    }
  }
  return TypeSwitch<Operation *, LogicalResult>(computeOp)
      .Default([&](auto op) {
        LDBG("Default Config");
        return setRootDefaultConfig(target, entryPointFn, computeOp);
      });
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//
LogicalResult initCudaGPULaunchConfig(FunctionOpInterface funcOp) {
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target)
    return funcOp.emitError("missing GPU target in #hal.executable.target");

  auto exportOp = getEntryPoint(funcOp);
  if (!getTranslationInfo(funcOp) && exportOp) {
    // If no translation info set, first check whether we already have
    // workgroup count set--it's a "contract" to indicate that we should
    // bypass all tiling and distribution to go down just the most basic
    // lowering flow.
    if (Block *body = exportOp->getWorkgroupCountBody()) {
      auto retOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
      // For scalar dispatch cases--using just one thread of one workgroup.
      auto isOne = [](Value value) { return matchPattern(value, m_One()); };
      if (llvm::all_of(retOp.getOperands(), isOne)) {
        SmallVector<int64_t, 3> workgroupSize = {1, 1, 1};
        auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
            funcOp.getContext(), CodeGenPipeline::LLVMGPUBaseLowering,
            workgroupSize);
        if (failed(setTranslationInfo(funcOp, translationInfo))) {
          return failure();
        }
        return success();
      }
    }
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  Operation *rootOperation = nullptr;

  // Find the root operation. linalg.generic, linalg.fill, tensor.pack,
  // tensor.unpack, and scatter are not root operations if there are other
  // compute operations present. Also, construct a set of generic ops that
  // are to be skipped. These generic ops that are used to compute scatter
  // indices are not root operations.
  llvm::SmallDenseSet<Operation *, 4> genericToSkip;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (!isa<linalg::GenericOp, linalg::FillOp, IREE::LinalgExt::ScatterOp,
             tensor::PackOp, tensor::UnPackOp>(op)) {
      rootOperation = op;
      break;
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      // linalg.generic with `reduction` iterator types are roots as well.
      if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
        rootOperation = op;
        break;
      }
    }

    if (auto scatterOp = dyn_cast<IREE::LinalgExt::ScatterOp>(op)) {
      Value indices = scatterOp.getIndices();
      if (!indices.getDefiningOp()) {
        continue;
      }

      // Mark scatter's backward slices(inclusive) as to skip.
      BackwardSliceOptions options;
      options.inclusive = true;
      SetVector<Operation *> slices;
      getBackwardSlice(indices, &slices, options);
      genericToSkip.insert(slices.begin(), slices.end());
    }
  }

  // Generic ops take priority over pack, unpack, scatter, and fill ops as the
  // root op.
  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<linalg::GenericOp>(op) && !genericToSkip.contains(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  // Pack and unpack ops take priority over scatter and fill ops as the root op.
  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<tensor::PackOp, tensor::UnPackOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<IREE::LinalgExt::ScatterOp, linalg::FillOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    // No root operation found, set it to none.
    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        funcOp.getContext(), CodeGenPipeline::None);
    if (failed(setTranslationInfo(funcOp, translationInfo))) {
      return failure();
    }
    return success();
  }

  if (failed(setRootConfig(target, funcOp, rootOperation)))
    return funcOp.emitOpError("failed to set root config");

  return success();
}

} // namespace mlir::iree_compiler
