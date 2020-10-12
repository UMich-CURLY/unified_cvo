
cd build && make -j6 && cd .. && \
for i in 05 04
#for i in  00 01 02 03 04 05 06 07 08 09 10
do
    #gdb -ex run --args \
	    ./build/bin/cvo_align_gpu_lidar_normal /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_normal.yaml \
                                       cvo_geometric_normal_$i.txt 0 200000
    
done
