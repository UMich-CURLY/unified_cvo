<<<<<<< Updated upstream
cd build
make -j
cd ..

for i in 05 07 00 02 06 08 09 
do
mkdir -p ~/media/sdg1/rzh/kitti_lidar/dataset/sequences/$i/pcd_semantic_downsampled
./build/bin/kitti_lidar_to_pcd ~/media/sdg1/rzh/kitti_lidar/dataset/sequences/$i ~/media/sdg1/rzh/kitti_lidar/dataset/sequences/$i/pcd_semantic_downsampled/ 0.2 1
done


=======

cd build
make -j12
cd ..

seq=05
rm -rf kitti_lidar_pcd_lidar_frame_$seq
mkdir -p kitti_lidar_pcd_lidar_frame_$seq #lidar frame lidar observations
./build/bin/kitti_lidar_to_pcd /home/`whoami`/media/Samsung_T5/kitti_lidar/dataset/sequences/${seq}/ kitti_lidar_pcd_lidar_frame_$seq 0.1 
>>>>>>> Stashed changes
