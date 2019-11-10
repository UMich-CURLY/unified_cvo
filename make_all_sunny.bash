#!/bin/bash
mkdir -p build_intel
cd build_intel
cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DGTSAM_DIR=/home/rzh/thirdparty/gtsam/build_sunny -DSophus_DIR=/home/rzh/thirdparty/Sophus-1.0.0
make -j 
