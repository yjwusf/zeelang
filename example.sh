#!/bin/bash
# Example script demonstrating the Zee MLIR dialect

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZEE_OPT="$SCRIPT_DIR/build/bin/zee-opt"
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
