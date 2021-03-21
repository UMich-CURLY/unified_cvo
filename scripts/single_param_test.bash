#export CUDA_VISIBLE_DEVICES=1

cd build && make -j && cd .. && \
#gdb -ex=r --args \
    #./build/bin/cvo_single_test  /home/v9999/media/seagate_2t/kitti/stereo/  cvo_params/cvo_semantic_params_img_gpu0.yaml single_tests_backup  1
    ./build/bin/cvo_single_semantic_test  /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/  cvo_params/cvo_geometric_params_img_gpu0.yaml single_tests  0
