
cd build && make -j4 && cd .. && \
for i in 05
#for i in  00 01 02 03 04 05 06 07 08 09 10
do
    echo ""
    echo "/********************** New Iteration *************************/"

    #gdb -ex run --args \
	    ./build/bin/cvo_align_gpu_lidar_raw_cov /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_geometric_params_dense_kernel.yaml \
                                       cvo_geometric_cov_$i.txt 0 20000
done
