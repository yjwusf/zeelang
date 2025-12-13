#!/bin/bash
set -e
cmake --preset default
cd build && make -j$(nproc)
