#include "CudaKernelConfig.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace cellar::target::cuda {

#define GEN_PASS_DEF_CUDAGPUSELECTLOWERINGSTRATEGYPASS
#include "compiler/plugins/target/CellarCUDA/CodeGen/Passes.h.inc" 

namespace {
/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class CudaGPUSelectLoweringStrategyPass final
    : public impl::CudaGPUSelectLoweringStrategyPassBase<
    CudaGPUSelectLoweringStrategyPass> {
public:
    using impl::CudaGPUSelectLoweringStrategyPassBase<
    CudaGPUSelectLoweringStrategyPass>::CudaGPUSelectLoweringStrategyPassBase;

    void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::GPU::IREEGPUDialect>();
    }

    void runOnOperation() override;
};

} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(FunctionOpInterface funcOp,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            ArrayRef<int64_t> workgroupSize, F verificationFn) {
    auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig =
        getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(op);
        if (!loweringConfig){
            return WalkResult::advance();
        }
        return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
    });
    return failure(walkResult.wasInterrupted());
}

static LogicalResult
verifyEntryPoint(FunctionOpInterface funcOp,
                    IREE::Codegen::TranslationInfoAttr translationInfo) {
    std::optional<SmallVector<int64_t>> workgroupSize = getWorkgroupSize(funcOp);
    if (!workgroupSize) {
        return funcOp->emitOpError(
            "failed to get workgroup size needed for verification");
    }

    return verifyLoweringConfiguration(
        funcOp, translationInfo, workgroupSize.value(), verifyGPUMatmulPipeline);
    return success();
}

void CudaGPUSelectLoweringStrategyPass::runOnOperation() {
    auto moduleOp = getOperation();
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
        if (mlir::failed(initCudaGPULaunchConfig(funcOp))) {
            return signalPassFailure();
        }

        IREE::Codegen::TranslationInfoAttr translationInfo =
            getTranslationInfo(funcOp);
        if (!translationInfo) {
            // Dont do anything if translation info is not set.
            return;
        }

        // Verify the properties of each entry point based on the target pipeline.
        if (mlir::failed(verifyEntryPoint(funcOp, translationInfo))) {
            return signalPassFailure();
        }
    }
}
    
}