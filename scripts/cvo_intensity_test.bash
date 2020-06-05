
cd build && make -j && cd .. && \
for i in 06
#for i in 00 02 03 04 05 06 07 08 09 10
do
	gdb -ex=r --args \
    ./build/bin/cvo_align_gpu_lidar_intensity_raw /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_intensity_params_gpu.yaml \
                                       cvo_intensity_$i.txt 0 20000
 
done
