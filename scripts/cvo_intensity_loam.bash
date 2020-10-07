
cd build && make -j6 && cd .. && \
for i in 10 04 06 
#for i in 04 05 03 10 06 00 01 02 07 08 09
do
    #	gdb -ex=r --args \
    ./build/bin/cvo_align_gpu_lidar_intensity_loam_raw /home/rayzhang/data/kitti_lidar/dataset/sequences/$i cvo_params/cvo_intensity_params_loam.yaml \
                                       cvo_intensity_loam_$i.txt 0 20000
 
done
