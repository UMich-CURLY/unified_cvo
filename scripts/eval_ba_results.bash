
class=$1
date=$2

cd devkit/cpp
g++ -g -o evaluate_odometry evaluate_odometry.cpp matrix.cpp
cd ../..



for i in 00 01 02 03 04 05 06 07 08 09 10
do
	#cp kitti_${class}_${i}_${date}/ba_kitti
	echo "=================================="
	result_dir=kitti_${class}_${i}_${date}/
	mkdir -p $result_dir/$i/
	#cp $result_dir/groundtruth_kitti.txt $result_dir/$i/
	cp ground_truth/kitti/lidar/${i}.txt $result_dir/$i/poses.txt
	cp $result_dir/ba_lidar.txt $result_dir/$i/
	cp $result_dir/pgo.txt $result_dir/$i/
	cp $result_dir/tracking_lidar.txt $result_dir/$i/tracking.txt
	cp results/balm24/balm_hba_results_2_2_24/results/${i}/hba_full.kitti ${result_dir}/${i}/hba.txt
	cp results/balm24/balm_hba_results_2_2_24/results/${i}/balm_full.kitti ${result_dir}/${i}/balm.txt
	cp results/balm24/balm_hba_results_2_2_24/results/${i}/init.kitti ${result_dir}/${i}/init.txt

	echo "seq $i "
	echo "ba for ${result_dir}:"
	 echo "	./devkit/cpp/evaluate_odometry $i ${result_dir} gt.txt $result_dir  ba.txt"
	 ./devkit/cpp/evaluate_odometry $i ${result_dir} poses.txt $result_dir  ba_lidar.txt
	 #./devkit/cpp/evaluate_odometry $i ${result_dir} gt.txt $result_dir  ba_kitti.txt
	echo "pgo for ${result_dir}: "
	 ./devkit/cpp/evaluate_odometry $i ${result_dir} poses.txt $result_dir  pgo.txt
	 #./devkit/cpp/evaluate_odometry $i ${result_dir} gt.txt $result_dir  pgo_kitti.txt
	echo "tracking for ${result_dir}: "
	 ./devkit/cpp/evaluate_odometry $i ${result_dir} poses.txt $result_dir  tracking.txt
	echo "hba for ${result_dir}: "
	 ./devkit/cpp/evaluate_odometry $i ${result_dir} poses.txt $result_dir  hba.txt
	echo "balm for ${result_dir}: "
	 ./devkit/cpp/evaluate_odometry $i ${result_dir} poses.txt $result_dir  balm.txt
	echo "balm for ${result_dir}: "
	 ./devkit/cpp/evaluate_odometry $i ${result_dir} poses.txt $result_dir  init.txt
	#echo "tracking for ${result_dir}: "
	# ./devkit/cpp/evaluate_odometry $i ${result_dir} groundtruth_kitti.txt $result_dir   odom${i}original_kitti.txt
	 #./devkit/cpp/evaluate_odometry $i ${result_dir} gt.txt $result_dir  odom${i}original_kitti.txt

done
