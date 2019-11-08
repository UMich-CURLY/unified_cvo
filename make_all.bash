#!/bin/bash
mkdir -p build
cd build
cmake .. -DSophus_DIR=/home/rayzhang/code/thirdparty/Sophus-1.0.0
make -j


