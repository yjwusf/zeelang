// RUN: zee-opt %s -convert-stablehlo-to-zee | FileCheck %s

// CHECK-LABEL: func.func @test_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>)
func.func @test_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RESULT:.*]] = zee.add %[[ARG0]], %[[ARG1]] : tensor<4xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_subtract
func.func @test_subtract(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: zee.subtract
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_multiply
func.func @test_multiply(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: zee.multiply
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @test_divide
func.func @test_divide(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.divide
  %0 = stablehlo.divide %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_maximum
func.func @test_maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.maximum
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_minimum
func.func @test_minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.minimum
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_abs
func.func @test_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.abs
  %0 = stablehlo.abs %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_negate
func.func @test_negate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.negate
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_exponential
func.func @test_exponential(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.exponential
  %0 = stablehlo.exponential %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_log
func.func @test_log(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: zee.log
  %0 = stablehlo.log %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_constant
func.func @test_constant() -> tensor<2x2xf32> {
  // CHECK: zee.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]>
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_composite
// CHECK-SAME: (%[[A:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>)
func.func @test_composite(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[ADD:.*]] = zee.add %[[A]], %[[B]]
  %0 = stablehlo.add %a, %b : tensor<4xf32>
  // CHECK: %[[MUL:.*]] = zee.multiply %[[ADD]], %[[A]]
  %1 = stablehlo.multiply %0, %a : tensor<4xf32>
  // CHECK: %[[NEG:.*]] = zee.negate %[[MUL]]
  %2 = stablehlo.negate %1 : tensor<4xf32>
  // CHECK: return %[[NEG]]
  return %2 : tensor<4xf32>
}
