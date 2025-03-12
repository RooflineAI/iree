namespace mlir {
    class OpPassManager;
}

namespace cellar::target::cuda{
/// Populates passes needed to preprocess and select the translation strategy.
void buildCudaGPUCodegenConfigurationPassPipeline(
    mlir::OpPassManager &variantPassManagery);
}