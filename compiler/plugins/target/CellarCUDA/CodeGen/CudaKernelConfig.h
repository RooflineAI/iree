#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace cellar::target::cuda {
    mlir::LogicalResult initCudaGPULaunchConfig(mlir::FunctionOpInterface funcOp);
}