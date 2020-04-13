# correct
#data_seq=host-a012-lidar0-1237329862198269106-1237329887099105436
#data_seq=host-a004-lidar0-1232817645198462196-1232817670098101226

# wrong starting at frame 44-45
#data_seq=host-a004-lidar0-1233011743198634026-1233011768099043756
# wrong starting at frame 104
data_seq=host-a004-lidar0-1232815252198642176-1232815277099387856
# wrong starting at frame 40
#data_seq=host-a004-lidar0-1232825386198046196-1232825411098056856

mkdir -p results/lyft/cvo_intensity_lidar/

cd build && make -j && cd .. && \

    ./build/bin/cvo_align_gpu_lidar_lyft /home/rayzhang/code/docker_home/media/Samsung_T5/lyft/v1.02-train/lidar/$data_seq \
                                         cvo_params/cvo_intensity_lyft_params_gpu.yaml \
                                         results/lyft/cvo_intensity_lidar/$data_seq.txt 100 2 
