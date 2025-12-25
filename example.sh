#!/bin/bash
# Example script demonstrating the Zee MLIR dialect

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZEE_OPT="$SCRIPT_DIR/build/bin/zee-opt"
ZEE_JIT="$SCRIPT_DIR/build/bin/zee-jit"
EXAMPLES_DIR="$SCRIPT_DIR/examples"

echo "=== Example 1: Parse and print zee.foo operation ==="
$ZEE_OPT "$EXAMPLES_DIR/test_foo.mlir"

echo ""
echo "=== Example 2: Run zee-switch-bar-foo pass (renames 'bar' to 'foo') ==="
$ZEE_OPT --zee-switch-bar-foo "$EXAMPLES_DIR/test_bar.mlir"

echo ""
echo "=== Example 3: Lower zee.foo to arith.addi ==="
$ZEE_OPT --convert-zee-to-arith "$EXAMPLES_DIR/test_foo.mlir"

echo ""
echo "=== Example 4: Roundtrip zee operations through parser ==="
$ZEE_OPT "$EXAMPLES_DIR/test_roundtrip.mlir" | $ZEE_OPT

echo ""
echo "=== Example 5: Full lowering to LLVM dialect ==="
$ZEE_OPT --convert-zee-to-arith --lower-to-llvm "$EXAMPLES_DIR/lower_to_llvm.mlir"

echo ""
echo "=== Example 6: JIT execution - simple computation (returns 42) ==="
$ZEE_JIT "$EXAMPLES_DIR/jit_simple.mlir"
RESULT=$?
echo "Return code: $RESULT (expected: 42)"

echo ""
echo "=== Example 7: JIT execution with print ==="
echo "Running jit_print.mlir (should print 42):"
$ZEE_JIT "$EXAMPLES_DIR/jit_print.mlir"

echo ""
echo "=== Example 8: JIT computation pipeline ==="
echo "Running jit_computation.mlir (should print 10, 20, 42):"
$ZEE_JIT "$EXAMPLES_DIR/jit_computation.mlir"

echo ""
echo "=== Example 9: JIT with LLVM IR dump ==="
echo "Running with --dump-llvm-ir flag:"
$ZEE_JIT --dump-llvm-ir "$EXAMPLES_DIR/jit_simple.mlir" 2>&1 | head -50
