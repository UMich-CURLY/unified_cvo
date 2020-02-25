cd build && make -j && cd ..

./build/bin/acvo_align_semantic_gpu_raw_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/04 cvo_params/acvo_params_gpu.txt \
                                       accum04.txt 0 800 


./build/bin/acvo_align_semantic_gpu_raw_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/04 cvo_params/acvo_params_gpu_2.4.txt \
                                       accum04_2.4.txt 0 800 

./build/bin/acvo_align_gpu_semantic_raw_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/04 cvo_params/acvo_params_gpu_1.6.txt \
                                       accum04_1.6.txt 0 800



./build/bin/acvo_align_gpu_semantic_raw_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/04 cvo_params/acvo_params_gpu_3.2.txt \
                                       accum04_3.2.txt 0 800
