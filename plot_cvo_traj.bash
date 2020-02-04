dataset="05"
file_name="cvo_f2f_tracking_"$dataset".txt"

# file_name="cvo_kf_tracking.txt"

evo_traj kitti --ref /home/cel/PERL/datasets/kitti_dataset/sequences/$dataset/groundtruth.txt  /home/cel/PERL/Algorithms/outdoor_cvo_lidar/results/$file_name -p
