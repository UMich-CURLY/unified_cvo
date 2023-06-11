date=$1

rm -rf cvo_align_lidar_$date
mkdir -p cvo_align_lidar_$date
cd build && make -j6 && cd .. && \
#for i in 10 05 03   
for i in  05 02 00 06 07 08 09 10
do
    #	gdb -ex=r --args \

        
        ./build/bin/cvo_align_gpu_lidar_intensity /home/rayzhang/media/Samsung_T5/kitti_lidar/dataset/sequences/$i cvo_params/cvo_kitti_lidar_params.yaml ${i}.txt 0 100000

        mv ${i}.txt cvo_align_lidar_$date
                                      # cvo_intensity_lidar_$i.txt 0 300002
 
done
