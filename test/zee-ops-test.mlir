// RUN: zee-opt %s | zee-opt | FileCheck %s

module {
  // CHECK-LABEL: func @test_foo
  func.func @test_foo(%arg0: i32) -> i32 {
    // CHECK: zee.foo
    %0 = zee.foo %arg0 : i32
    return %0 : i32
  }
}
