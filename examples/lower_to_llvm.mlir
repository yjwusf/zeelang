// Example showing the full lowering pipeline with zee-opt
// This can be used to inspect the LLVM dialect output
//
// Run with:
//   ./build/bin/zee-opt examples/lower_to_llvm.mlir \
//     --convert-zee-to-arith --lower-to-llvm

module {
  func.func @compute(%arg0: i32) -> i32 {
    %c10 = arith.constant 10 : i32
    %sum = arith.addi %arg0, %c10 : i32
    %result = zee.foo %sum : i32
    return %result : i32
  }
}
