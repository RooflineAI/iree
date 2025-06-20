// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(torch-iree-func-conversion)" --verify-diagnostics %s

// We do not support returning a mutable tensor from a function.
// It is unclear if these can be generated from current torch tooling. If it
// ever becomes a problem, something can be implemented.
builtin.module @mutable_input_overwrite_return {
// expected-error @+1 {{mutable tensor arguments are not supported: '!torch.tensor<[5,4],f32>'}}
func.func @main(%arg0: !torch.tensor<[5,4],f32>) -> (!torch.tensor<[5,4],f32>) {
  %0 = torch.copy.to_vtensor %arg0 : !torch.vtensor<[5,4],f32>
  %1 = torch.operator "mutate_inplace"(%0) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  torch.overwrite.tensor.contents %1 overwrites %arg0 : !torch.vtensor<[5,4],f32>, !torch.tensor<[5,4],f32>
  return %arg0 : !torch.tensor<[5,4],f32>
}
}
