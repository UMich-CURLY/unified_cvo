
cd build && make -j && cd .. && \
<<<<<<< HEAD
#for i in 04
export CUDA_VISIBLE_DEVICES=1
for i in 00 01 02 03 04 05 06 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_semantic_raw /home/rzh/media/sda1/ray/datasets/kitti_lidar/dataset/sequences/$i cvo_params/cvo_semantic_params_gpu.txt \
                                       cvo_semantic_$i.txt 0   10000
=======
for i in 05

#for i in 00 01 02 03 04 05 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_semantic_raw /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_semantic_params_gpu.txt \
                                       cvo_semantic_$i.txt 1772  2
>>>>>>> origin/velocity_ell

done
