
cd build && make -j && cd .. && \
for i in 04
#for i in 00 01 02 03 04 05 07 08 09 10
do
    ./build/bin/acvo_align_semantic_gpu_raw_lidar /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/acvo_semantic_params_gpu.txt \
                                       acvo_semantic_lidar_$i.txt 0   10000

done
