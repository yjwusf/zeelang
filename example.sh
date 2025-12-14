#!/bin/bash
# Example script demonstrating the Zee MLIR dialect

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZEE_OPT="$SCRIPT_DIR/build/bin/zee-opt"

echo "=== Example 1: Parse and print zee.foo operation ==="
$ZEE_OPT <<EOF
module {
  func.func @test_foo(%arg0: i32) -> i32 {
    %0 = zee.foo %arg0 : i32
    return %0 : i32
  }
}
EOF

echo ""
echo "=== Example 2: Run zee-switch-bar-foo pass (renames 'bar' to 'foo') ==="
$ZEE_OPT --zee-switch-bar-foo <<EOF
func.func @bar() {
  return
}
EOF

echo ""
echo "=== Example 3: Roundtrip zee operations through parser ==="
$ZEE_OPT <<EOF | $ZEE_OPT
module {
  func.func @test_roundtrip(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zee.foo %arg0 : i32
    %1 = zee.foo %arg1 : i32
    return %0 : i32
  }
}
EOF
