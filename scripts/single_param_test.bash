cd build && make -j && cd ..

#./build/bin/acvo_align_semantic_gpu_raw_img /home/rayzhang/data/kitti/04 cvo_params/acvo_semantic_params_gpu.txt \
#                                       accum04_semantic_1.1.txt 0 270


./build/bin/acvo_align_semantic_gpu_raw_img /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/05_raw_imgs/05/ cvo_params/acvo_semantic_params_gpu.txt \
                                       acvo_semantic_params_gpu_1.6.txt 0 3000
#./build/bin/acvo_align_semantic_gpu_raw_img /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/05_raw_imgs/05/ cvo_params/acvo_semantic_params_gpu_0.6.txt \
#                                       accum05_semantic_0.5.txt 0 3000

#./build/bin/acvo_align_semantic_gpu_raw_img /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/05_raw_imgs/05/ cvo_params/acvo_semantic_params_gpu_1.0.txt \
#                                       accum05_semantic_1.0.txt 0 3000
