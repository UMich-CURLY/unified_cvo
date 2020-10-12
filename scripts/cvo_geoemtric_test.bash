
export CUDA_VISIBLE_DEVICES=1
cd build && make -j && cd .. && \
for i in 08 
#for i in 00 01 02 03 04 05
do
    ./build/bin/cvo_align_gpu_lidar_raw /home/v9999/media/seagate_2t/kitti/lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_gpu.yaml \
                                       cvo_geometric_$i.txt  0 100000
done
