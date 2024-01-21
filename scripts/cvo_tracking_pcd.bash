cd build
make -j
cd ..

mkdir -p build_debug
cd build_debug
make -j12
cd ..

gdb -ex run --args \
./build_debug/bin/cvo_align_gpu_pcd  ../cassie_fxb/segmented_pcd/ cvo_params/cvo_tracking_cassie.yaml tracking_${folder}.txt 0 1261 1
