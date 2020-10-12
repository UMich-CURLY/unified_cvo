
cd build && make -j6 && cd .. && \
for i in 04 10 
#for i in   00 01 02 03 04 07 08 09
do
    #gdb -ex run --args \
	    ./build/bin/cvo_align_gpu_lidar_loam /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_loam.yaml \
                                       cvo_geometric_loam_$i.txt 0 30000
    
done
