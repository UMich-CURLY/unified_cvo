
cd build && make -j && cd .. && \

./build/bin/acvo_geometric_gpu_raw_lidar /home/rayzhang/data/03 cvo_params/acvo_geometric_params_gpu.txt \
                                       acvo_geometric_03.txt 0 1000
