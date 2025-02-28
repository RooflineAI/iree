#include "CudaGPUCodeGenPipeline.h"
#include "Passes.h"

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
    void buildCudaGPUCodegenConfigurationPassPipelineImpl(
        OpPassManager &modulePassManager) {
      {
        FunctionLikeNest funcPassManager(modulePassManager);
        funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
        addCommonTargetExecutablePreprocessingPasses(funcPassManager);
        // This materializes into 'nop' in the absence of pad encoding layout
        // attributes.
        addEncodingToPaddingPasses(funcPassManager);
        funcPassManager.addPass(createBlockDynamicDimensionsPass);
        funcPassManager.addPass(createConfigTrackingCanonicalizerPass);
        funcPassManager.addPass(createCSEPass);
      }
      modulePassManager.addPass(createMaterializeTuningSpecsPass());
      modulePassManager.addPass(createMaterializeUserConfigsPass());
      modulePassManager.addPass(cellar::target::cuda::createCudaGPUSelectLoweringStrategyPass());
    }
}

namespace cellar::target::cuda{
    /// Populates passes needed to preprocess and select the translation strategy.
    void buildCudaGPUCodegenConfigurationPassPipeline(
        mlir::OpPassManager &variantPassManagery){
            buildCudaGPUCodegenConfigurationPassPipelineImpl(
                variantPassManagery.nest<ModuleOp>());
        }
}