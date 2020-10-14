
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=1
cd build && make -j && cd .. && \
for i in 08 
#for i in 00 01 02 03 04 05
do
    ./build/bin/cvo_align_gpu_lidar_raw /home/v9999/media/seagate_2t/kitti/lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_gpu.yaml \
                                       cvo_geometric_$i.txt  0 100000
=======
cd build && make -j6 && cd .. 
export CUDA_VISIBLE_DEVICES=0
for i in 05
#for i in 00 01 02 03 04 05 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_raw /home/rzh/media/sda1/ray/datasets/kitti_lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_gpu.yaml \
                                       cvo_geometric_raw_$i.txt 0   10000
>>>>>>> origin/sunny
done
