dataset="01"
file_name="cvo_f2f_tracking_"$dataset".txt"

# file_name="cvo_kf_tracking.txt"

evo_traj kitti --ref /media/justin/LaCie/data/kitti/sequences/$dataset/groundtruth.txt  ~/research/outdoor_cvo/$file_name -p
