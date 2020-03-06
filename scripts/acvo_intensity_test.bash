
cd build && make -j && cd .. && \
for i in 04
#for i in 00 01 02 03 04 05 07 08 09 10
do
    ./build/bin/acvo_intensity_gpu_raw_lidar /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/acvo_intensity_params_gpu.txt \
                                       acvo_intensity_lidar_$i.txt  0 2

done
