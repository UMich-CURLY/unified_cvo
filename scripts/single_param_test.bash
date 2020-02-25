cd build && make -j && cd ..

./build/bin/acvo_align_gpu_raw_img /home/rayzhang/data/kitti/04 cvo_params/acvo_params_gpu.txt \
                                       accum04.txt 0 800 



./build/bin/acvo_align_gpu_raw_img /home/rayzhang/data/kitti/04 cvo_params/acvo_params_gpu_2.4.txt \
                                       accum04_2.4.txt 0 800



./build/bin/acvo_align_gpu_raw_img /home/rayzhang/data/kitti/04 cvo_params/acvo_params_gpu_3.2.txt \
                                       accum04_3.2.txt 0 800
