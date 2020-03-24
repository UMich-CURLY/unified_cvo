
cd build && make -j && cd .. && \
for i in 05
#for i in 05 00 02  06 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_raw /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_gpu.yaml \
                                       cvo_geometric_$i.txt  0 820
done
