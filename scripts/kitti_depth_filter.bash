
seq=05
frame=117

direll=0.25
normalell=0.04

cd build
make -j
cd ..
#valgrind --leak-check=yes \
#cuda-gdb -ex run --args \
	./build/bin/kitti_depth_filtering  ../media/Samsung_T5/kitti/${seq}/ cvo_params/cvo_intensity_params_img_gpu0.yaml results/cvo_intensity_img_gpu0_oct25_best/${seq}.txt $frame 5 $direll $normalell
