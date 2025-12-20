#!/bin/bash
set -e
cmake --preset default -DMLIR_DIR=/opt/homebrew/Cellar/llvm@20/20.1.8/lib/cmake/mlir

# Use nproc on Linux, sysctl on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
  NPROC=$(sysctl -n hw.ncpu)
else
  NPROC=$(nproc)
fi

cd build && make -j$NPROC
