

results_dir=$1/

cd devkit/cpp
g++ -g -o evaluate_odometry evaluate_odometry.cpp matrix.cpp
cd ../..

for seq in 00 01 02 03 04 05 06 07 08 09 10
do
results_file_name=$seq.txt
gt_file_name=$seq.gt.txt

  ./devkit/cpp/evaluate_odometry $seq $results_dir $gt_file_name $results_dir      $results_file_name
done


