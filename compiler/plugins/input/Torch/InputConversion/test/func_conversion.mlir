// RUN: iree-opt --pass-pipeline="builtin.module(torch-iree-func-conversion)" --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// Canonical test of the immutable input->compute->return case.
// CHECK-LABEL: @immutable_import_export
//       CHECK: util.func public @main(
//  CHECK-SAME:     %arg0: tensor<4x5xi32>, %arg1: tensor<5x4xf32>) ->
//  CHECK-SAME:     (tensor<4x5xi32>, tensor<5x4xf32>)
//   CHECK-DAG:   %[[TORCH_ARG0:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<4x5xi32> -> !torch.vtensor<[4,5],si32>
//   CHECK-DAG:   %[[TORCH_ARG1:.+]] = torch_c.from_builtin_tensor %arg1 : tensor<5x4xf32> -> !torch.vtensor<[5,4],f32>
//   CHECK-DAG:   %[[TORCH_RESULT0:.+]] = torch.operator "foobar0"(%[[TORCH_ARG0]])
//   CHECK-DAG:   %[[TORCH_RESULT1:.+]] = torch.operator "foobar1"(%[[TORCH_ARG1]])
//   CHECK-DAG:   %[[TENSOR_RESULT0:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT0]]
//   CHECK-DAG:   %[[TENSOR_RESULT1:.+]] = torch_c.to_builtin_tensor %[[TORCH_RESULT1]]
//       CHECK:   util.return %[[TENSOR_RESULT0]], %[[TENSOR_RESULT1]]
builtin.module @immutable_import_export {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>, %arg1: !torch.vtensor<[5,4],f32>)
    -> (!torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>) {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  %1 = torch.operator "foobar1"(%arg1) : (!torch.vtensor<[5,4],f32>) -> !torch.vtensor<[5,4],f32>
  return %0, %1 : !torch.vtensor<[4,5],si32>, !torch.vtensor<[5,4],f32>
}
}

// -----
// CHECK-LABEL: @return_immutable_arg
// CHECK: util.func public @main(
// CHECK-SAME: %arg0: tensor<4x5xi32>) -> tensor<4x5xi32>
// CHECK: util.return %arg0
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_reflection
//      CHECK: util.func public @main(
// CHECK-SAME:   iree.reflection = {some.attr = 4 : index}
builtin.module @retained_attribute_reflection {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  attributes {
    iree.reflection = {
      some.attr = 4 : index
    }
  }
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_ignored
//      CHECK: util.func public @main(
//  CHECK-NOT: iree.nonretained
builtin.module @retained_attribute_ignored {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  attributes {
    iree.nonretained = "dummy"
  }
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @retained_attribute_noinline
//      CHECK: util.func public @main(
// CHECK-SAME:   noinline
builtin.module @retained_attribute_noinline {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
  attributes {
    noinline
  }
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @private_visibility
// CHECK: util.func private @main
builtin.module @private_visibility {
func.func private @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// CHECK-LABEL: @tied_operand
// CHECK: util.func public @main(%arg0: tensor<4x5xi32>) -> (%arg0 {iree.abi.tied = 0 : i64})
// CHECK: util.return %arg0
builtin.module @tied_operand {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) ->
  (!torch.vtensor<[4,5],si32> {iree.abi.tied = 0})
{
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}

// -----
// Verify that dynamic dimensions work.
// CHECK-LABEL: @immutable_import_export_dynamic
// CHECK: util.func public @main(%arg0: tensor<4x?xi32>, %arg1: tensor<?x4xf32>)
builtin.module @immutable_import_export_dynamic {
func.func @main(%arg0: !torch.vtensor<[4,?],si32>, %arg1: !torch.vtensor<[?,4],f32>)
    -> (!torch.vtensor<[4,?],si32>, !torch.vtensor<[?,4],f32>) {
  %0 = torch.operator "foobar0"(%arg0) : (!torch.vtensor<[4,?],si32>) -> !torch.vtensor<[4,?],si32>
  %1 = torch.operator "foobar1"(%arg1) : (!torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
  return %0, %1 : !torch.vtensor<[4,?],si32>, !torch.vtensor<[?,4],f32>
}
}

// -----
// CHECK-LABEL: @torch_bool_return
// CHECK: torch_c.to_i1
// CHECK: util.return {{.*}} : i1
module @torch_bool_return {
  func.func @main() -> !torch.bool {
    %0 = torch.operator "some.primitive"() : () -> !torch.bool
    return %0 : !torch.bool
  }
}

// -----
// CHECK-LABEL: @torch_int_return
// CHECK: torch_c.to_i64
// CHECK: util.return {{.*}} : i64
module @torch_int_return {
  func.func @main() -> !torch.int {
    %0 = torch.operator "some.primitive"() : () -> !torch.int
    return %0 : !torch.int
  }
}

// -----
// CHECK-LABEL: @torch_float_return
// CHECK: torch_c.to_f64
// CHECK: util.return {{.*}} : f64
module @torch_float_return {
  func.func @main() -> !torch.float {
    %0 = torch.operator "some.primitive"() : () -> !torch.float
    return %0 : !torch.float
  }
}

// -----
// CHECK-LABEL: @torch_generator_return
// CHECK: torch_c.generator_to_i64
// CHECK: util.return {{.*}} : i64
module @torch_generator_return {
  func.func @main() -> !torch.Generator {
    %0 = torch.operator "some.primitive"() : () -> !torch.Generator
    return %0 : !torch.Generator
  }
}

// -----
// CHECK-LABEL: @torch_bool_arg
// CHECK: torch_c.from_i1 %arg0
module @torch_bool_arg {
  func.func @main(%arg0 : !torch.bool) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.bool) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @torch_int_arg
// CHECK: torch_c.from_i64 %arg0
module @torch_int_arg {
  func.func @main(%arg0 : !torch.int) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.int) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @torch_float_arg
// CHECK: torch_c.from_f64 %arg0
module @torch_float_arg {
  func.func @main(%arg0 : !torch.float) -> (!torch.vtensor<[1],f32>) {
    %0 = torch.operator "some.primitive"(%arg0) : (!torch.float) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_index_arg
module @builtin_index_arg {
  func.func @main(%arg0 : index) -> (!torch.vtensor<[1],f32>) {
    %0 = "torch_test.operator"(%arg0) : (index) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_int_arg
module @builtin_int_arg {
  func.func @main(%arg0 : i32) -> (!torch.vtensor<[1],f32>) {
    %0 = "torch_test.operator"(%arg0) : (i32) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_float_arg
module @builtin_float_arg {
  func.func @main(%arg0 : f32) -> (!torch.vtensor<[1],f32>) {
    %0 = "torch_test.operator"(%arg0) : (f32) ->  (!torch.vtensor<[1],f32>)
    return %0 : !torch.vtensor<[1],f32>
  }
}

// -----
// CHECK-LABEL: @builtin_index_return
module @builtin_index_return {
  func.func @main() -> index {
    %0 = "torch_test.operator"() : () -> index
    return %0 : index
  }
}

// -----
// CHECK-LABEL: @builtin_int_return
module @builtin_int_return {
  func.func @main() -> i32 {
    %0 = "torch_test.operator"() : () -> i32
    return %0 : i32
  }
}

// -----
// CHECK-LABEL: @builtin_float_return
module @builtin_float_return {
  func.func @main() -> f32 {
    %0 = "torch_test.operator"() : () -> f32
    return %0 : f32
  }
}

// -----
// CHECK-LABEL: @device_affinity_preserved
// CHECK: util.func public @main(
// CHECK-SAME:   %arg0: tensor<2x2xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>},
// CHECK-SAME:   %arg1: tensor<2x2xf32> {iree.abi.affinity = #hal.device.affinity<@device_b>})
// CHECK-SAME:   -> (tensor<2x2xf32> {iree.abi.affinity = #hal.device.affinity<@device_c>})
module @device_affinity_preserved {
  func.func @main(%arg0: !torch.vtensor<[2,2],f32> {iree.abi.affinity = #hal.device.affinity<@device_a>},
                  %arg1: !torch.vtensor<[2,2],f32> {iree.abi.affinity = #hal.device.affinity<@device_b>})
                  -> (!torch.vtensor<[2,2],f32> {iree.abi.affinity = #hal.device.affinity<@device_c>})
                  attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.operator "some.operation"(%arg0, %arg1) : (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }
}
