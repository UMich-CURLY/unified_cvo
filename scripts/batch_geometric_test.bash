
cd build && make -j && cd ..
export CUDA_VISIBLE_DEVICES=1
for i in 00 01 02 03 04 05 06 07 08 09 10
do
        
    ./build/bin/acvo_geometric_gpu_raw_lidar /home/v9999/media/seagate_2t/kitti/lidar/dataset/sequences/$i cvo_params/acvo_geometric_params_gpu.txt \
                                             acvo_geometric_$i.txt 0 10000

done
