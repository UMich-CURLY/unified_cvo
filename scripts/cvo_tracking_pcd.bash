cd build
make -j6
cd ..

#mkdir -p build_debug
#cmake .. -DCMAKE_BUILD_TYPED=Debug
#cd build_debug
#make -j12
#cd ..

#gdb -ex run --args \
#./build/bin/cvo_align_gpu_pcd  ../cassie_fxb/segmented_pcd/ cvo_params/cvo_tracking_cassie.yaml ../cassie_fxb/tracking.txt 0 1260 1 #1260 1

./build/bin/cvo_align_gpu_pcd  ../cassie_wavefield_manual/segmented_pcd/ cvo_params/cvo_tracking_cassie.yaml ../cassie_wavefield_manual/tracking.txt 0 2569 1 1 #1260 1
