
cd build && make -j && cd .. && \
for i in 04
#for i in 00 01 02 03 04 05 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_raw /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_intensity_params_gpu.txt \
                                       cvo_intensity_lidar_$i.txt 0   10000

done
