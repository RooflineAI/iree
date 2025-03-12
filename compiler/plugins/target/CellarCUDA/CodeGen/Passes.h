#include "mlir/Pass/Pass.h"

namespace cellar::target::cuda{

#define GEN_PASS_DECL
#include "compiler/plugins/target/CellarCUDA/CodeGen/Passes.h.inc" // IWYU pragma: keep

} 