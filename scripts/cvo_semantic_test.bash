
cd build && make -j6 && cd .. && \
#for i in 04
export CUDA_VISIBLE_DEVICES=0
for i in 04 01 00 02 03 05 06 07 08 09 10
do
    ./build/bin/cvo_align_gpu_lidar_semantic_raw /home/v9999/media/seagate_2t/kitti/lidar/dataset/sequences/$i cvo_params/cvo_semantic_params_gpu.txt \
                                       cvo_semantic_$i.txt 0   10000

done
