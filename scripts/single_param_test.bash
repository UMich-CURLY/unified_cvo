#export CUDA_VISIBLE_DEVICES=1

cd build && make -j && cd .. && \


./build/bin/cvo_single_test  /home/v9999/media/seagate_2t/kitti/stereo/  cvo_params/cvo_geometric_params_img_gpu0.yaml single_tests  0
