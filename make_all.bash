#!/bin/bash
mkdir -p build
cd build
cmake .. -DSophus_DIR=/home/rayzhang/code/thirdparty/Sophus-1.0.0 #-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
make -j


