
export CUDA_VISIBLE_DEVICES=1
cd build && make -j && cd .. && \
for i in 04 
#for i in 00 01 02 03 04 05 06 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_raw /home/v9999/media/seagate_2t/kitti/lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_gpu.txt \
                                       cvo_geometric_$i.txt  0 2
done
