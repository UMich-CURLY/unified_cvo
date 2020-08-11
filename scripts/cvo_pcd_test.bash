
cd build && make -j6 && cd .. && \
    gdb -ex run --args \
	    ./build/bin/cvo_align_gpu_lidar_pcd /home/rayzhang/data/cassie_wavefield/ cvo_params/cvo_geometric_params_gpu.yaml \
                                       cassie.txt 0 100000 
    
