// Simple JIT example - returns 42
// This demonstrates full lowering: zee -> arith -> llvm -> native execution
//
// Run with: ./build/bin/zee-jit examples/jit_simple.mlir
// Expected output: exits with return code 42

module {
  func.func @main() -> i32 {
    %c21 = arith.constant 21 : i32
    // zee.foo doubles its input (adds to itself)
    %result = zee.foo %c21 : i32
    return %result : i32
  }
}
