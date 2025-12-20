module {
  func.func @test_foo(%arg0: i32) -> i32 {
    %c42 = arith.constant 42 : i32
    %1 = arith.addi %arg0, %c42 : i32
    %0 = zee.foo %1 : i32
    return %0 : i32
  }
}
