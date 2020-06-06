
cd build && make -j && cd .. && \
for i in 03
#for i in  00 01 02 03  06 07 08 09 10
do
	#gdb -ex=r --args \ 
    ./build/bin/cvo_align_gpu_lidar_loam /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_gpu.yaml \
                                       cvo_geometric_$i.txt 0 200000 
    
done
