# dataset="01"
# file_name="cvo_f2f_tracking_"$dataset"_08.txt"

# file_name="cvo_kf_tracking.txt"
for dataset in 05
do
    file_name="cvo_f2f_tracking_"$dataset"_lidar.txt"
    evo_traj kitti --ref /media/justin/LaCie/data/kitti/ground_truth/$dataset.txt  /home/justin/research/outdoor_cvo/results/$file_name /home/justin/research/outdoor_cvo/baselines/lidar/results_gicp/$dataset.txt  -p
    # evo_traj kitti --ref /media/justin/LaCie/data/kitti/sequences/$dataset/groundtruth.txt  /home/justin/research/outdoor_cvo/results/$file_name -p
    # evo_rpe kitti -p /media/justin/LaCie/data/kitti/sequences/$dataset/groundtruth.txt  /home/justin/research/outdoor_cvo/results/$file_name
done