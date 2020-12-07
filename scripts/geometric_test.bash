
cd build && make -j && cd .. 
export CUDA_VISIBLE_DEVICES=1

    for i in 00 01 
    ./build/bin/acvo_geometric_gpu_raw_lidar /home/rayzhang/data/kitti_lidar/dataset/sequences/00 cvo_params/acvo_geometric_params_gpu.txt \
                                       acvo_geometric_00.txt 3600   10000
			       done
