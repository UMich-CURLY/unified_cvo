
cd build && make -j6 && cd .. && \
for i in 04  
#for i in   00 01 02 03 04 07 08 09
do
    gdb -ex run --args \
	    ./build/bin/cvo_align_gpu_lidar_semantic_loam /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_semantic_params_loam.yaml \
                                       cvo_semantic_loam_$i.txt 0 2
    
done
