cd build
make -j
cd ..

for i in 05 07 00 02 06 08 09 
do
mkdir -p ~/media/sdg1/rzh/kitti_lidar/dataset/sequences/$i/pcd_semantic_downsampled
./build/bin/kitti_lidar_to_pcd ~/media/sdg1/rzh/kitti_lidar/dataset/sequences/$i ~/media/sdg1/rzh/kitti_lidar/dataset/sequences/$i/pcd_semantic_downsampled/ 0.2 1
done


