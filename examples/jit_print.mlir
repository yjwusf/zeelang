// JIT example with print - prints a value and returns 0
// Demonstrates the zee.print operation being lowered to printf
//
// Run with: ./build/bin/zee-jit examples/jit_print.mlir
// Expected output: prints "42" to stdout

module {
  func.func @main() -> i32 {
    %c21 = arith.constant 21 : i32
    // zee.foo doubles its input
    %doubled = zee.foo %c21 : i32
    // Print the result (should print 42)
    zee.print %doubled : i32
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
