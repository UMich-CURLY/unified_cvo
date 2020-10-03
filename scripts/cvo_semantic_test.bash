
cd build && make -j && cd .. && \
<<<<<<< HEAD
#for i in 04
export CUDA_VISIBLE_DEVICES=1
#for i in 04 01 00 02 03 05 
for i in 06 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_semantic_raw /home/v9999/media/seagate_2t/kitti/lidar/dataset/sequences/$i cvo_params/cvo_semantic_params_gpu.txt \
                                       cvo_semantic_$i.txt 0   10000
=======
for i in 02

#for i in 00 01 02 03 04 05 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_semantic_raw /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_semantic_params_gpu.yaml\
                                       cvo_semantic_$i.txt  0 2
>>>>>>> origin/indicator_merged

done
