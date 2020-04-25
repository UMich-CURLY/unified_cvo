gt_dir="/media/justin/LaCie/data/kitti/ground_truth/"

gicp_results_dir=/home/justin/research/outdoor_cvo/baselines/lidar/results_gicp/
ndt_results_dir=/home/justin/research/outdoor_cvo/baselines/lidar/results_ndt/
mc_results_dir=/home/justin/research/outdoor_cvo/baselines/lidar/results_mc/

results_dir=$1
seq=$2


cd devkit/cpp
g++-9 -g -o evaluate_odometry evaluate_odometry.cpp matrix.cpp
cd ../..


#for seq in 00 01 02 03 04 05 06 07 08 09 10
#do
echo $seq
    #for file in $results_dir"cvo_f2f_tracking_"$seq*; 
    #do
        #results_file_name="${file##*/}"
    #results_file_name=cvo_geometric_$1.txt
results_file_name=$seq.txt
gt_file_name=$seq.txt
echo $results_file_name

  ./devkit/cpp/evaluate_odometry $seq $gt_dir $gt_file_name $gicp_results_dir $results_file_name
  #./devkit/cpp/evaluate_odometry $seq $gt_dir $gt_file_name $ndt_results_dir  $results_file_name
  #./devkit/cpp/evaluate_odometry $seq $gt_dir $gt_file_name $mc_results_dir   $results_file_name
  ./devkit/cpp/evaluate_odometry $seq $gt_dir $gt_file_name $results_dir      $results_file_name
        # ./devkit/cpp/evaluate_odometry 05 /media/justin/LaCie/data/kitti/sequences/ groundtruth.txt ../../results/ cvo_f2f_tracking_05_08.txt
    #$done
#done


