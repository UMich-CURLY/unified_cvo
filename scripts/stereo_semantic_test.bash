


# run in outdoor_cvo root folder

cd build && make -j && cd ..

export CUDA_VISIBLE_DEVICES=0

for  i in 00 01 02 03 04 05
do
./build/bin/acvo_align_semantic_gpu /home/v9999/media/seagate_2t/kitti/stereo/$i/cvo_points_semantics cvo_params/acvo_semantic_params_gpu.txt acvo_semantic_stero_rela_$i.txt \
                                    acvo_semantic_stereo_$i.txt 0 10000

done
#./build/bin/acvo_align_semantic_gpu_raw_img /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/05_raw_imgs/05/ cvo_params/acvo_semantic_params_gpu_0.6.txt \
#                                       accum05_semantic_0.5.txt 0 3000

#./build/bin/acvo_align_semantic_gpu_raw_img /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/05_raw_imgs/05/ cvo_params/acvo_semantic_params_gpu_1.0.txt \
#                                       accum05_semantic_1.0.txt 0 3000
