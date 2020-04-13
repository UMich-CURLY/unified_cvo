# correct
#data_seq=host-a012-lidar0-1237329862198269106-1237329887099105436
#data_seq=host-a004-lidar0-1232817645198462196-1232817670098101226

# wrong starting at frame 44-45
#data_seq=host-a004-lidar0-1233011743198634026-1233011768099043756
# wrong starting at frame 104
#data_seq=host-a004-lidar0-1232815252198642176-1232815277099387856
# wrong starting at frame 40
data_seq=host-a004-lidar0-1232825386198046196-1232825411098056856


file_name="cvo_intensity_lidar_$data_seq.txt"

#evo_traj kitti /home/cel/PERL/code/DockerFolder/media/Samsung_T5/outdoor_cvo/results/lidar_intensity_result/lyft/$file_name -p

evo_traj kitti --ref /home/cel/PERL/code/DockerFolder/media/Samsung_T5/lyft/lyft_kitti/train/$data_seq/groundtruth.txt /home/cel/PERL/code/DockerFolder/media/Samsung_T5/outdoor_cvo/results/lidar_intensity_result/lyft/$file_name -p
