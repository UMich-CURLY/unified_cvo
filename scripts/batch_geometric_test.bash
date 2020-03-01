
cd build && make -j && cd .. 
for i in 00 01 02 03 04 05 06 07 08 09 10
do
        
    ./build/bin/acvo_geometric_gpu_raw_lidar /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/acvo_geometric_params_gpu.txt \
                                             results/lidar_geometric_result/acvo_geometric_$i.txt 0 10000

done

