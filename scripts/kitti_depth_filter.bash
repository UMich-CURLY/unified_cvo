
seq=05
frame=23

direll=1.0
normalell=0.1

cd build
make -j
cd ..
#valgrind --leak-check=yes \
cuda-gdb -ex run --args \
	./build/bin/kitti_depth_filtering  ../media/Samsung_T5/kitti_stereo/dataset/sequences/kitti/${seq}/ cvo_params/cvo_intensity_params_img_gpu0.yaml results/cvo_intensity_img_gpu0_oct25_best/${seq}.txt $frame 5 $direll $normalell
