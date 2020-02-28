
cd build && make -j && cd .. && \

    ./build/bin/acvo_geometric_gpu_raw_lidar /home/rayzhang/data/kitti_lidar/dataset/sequences/06 cvo_params/acvo_geometric_params_gpu.txt \
                                       acvo_geometric_06.txt 0   10000
