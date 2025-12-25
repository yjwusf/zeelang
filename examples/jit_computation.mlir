// JIT computation example - performs a sequence of operations
// Demonstrates multiple zee operations in a computation pipeline
//
// Run with: ./build/bin/zee-jit examples/jit_computation.mlir
// Expected output: prints the computation results

module {
  func.func @main() -> i32 {
    // Initial value
    %c5 = arith.constant 5 : i32

    // zee.foo doubles: 5 * 2 = 10
    %v1 = zee.foo %c5 : i32
    zee.print %v1 : i32

    // zee.foo again: 10 * 2 = 20
    %v2 = zee.foo %v1 : i32
    zee.print %v2 : i32

    // Add a constant: 20 + 22 = 42
    %c22 = arith.constant 22 : i32
    %v3 = arith.addi %v2, %c22 : i32
    zee.print %v3 : i32

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
