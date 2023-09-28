cd build/ && make -j && cd .. 

for i in gazebo_winter gazebo_summer
do	
./build/bin/cvo_align_gpu_lidar_ethz  ~/media/Samsung_T5/ethz/${i}/ cvo_params/cvo_ethz_lidar_params.yaml  ethz_${i}.txt 0 1000
done
