module {
  func.func @test_roundtrip(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zee.foo %arg0 : i32
    %1 = zee.foo %arg1 : i32
    return %0 : i32
  }
}
