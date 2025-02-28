#ifndef ROOF_CUDA_CODEGEN_GPU_PASSES
#define ROOF_CUDA_CODEGEN_GPU_PASSES

include "mlir/Pass/PassBase.td"

def CudaGPUSelectLoweringStrategyPass :
    Pass<"roof-cuda-gpu-select-lowering-strategy", "ModuleOp"> {
  let summary = "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the target variant";
}

#endif // ROOF_CUDA_CODEGEN_GPU_PASSES