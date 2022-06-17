#/bin/bash

seq=$1
frame=$2

cd build 
make -j 
cd ..
#gdb -ex run --args \
./build/bin/cvo_irls_kitti /home/rayzhang/media/Samsung_T5/kitti_stereo/dataset/sequences/kitti/05/ cvo_params/cvo_intensity_params_irls_kitti.yaml /home/rayzhang/dsm/kitti_color_${seq}/${frame}_graph.txt results/${seq}.txt  ground_truth/${seq}.txt 
#echo "Evaluation: "
#echo "before BA ate:"
#python3 evaluate_ate_scale/evaluate_ate_scale.py --plot before_traj.png --scale 1  --verbose gt_poses.txt cvo_track_poses.txt
#python3 ~/.local/lib/python3.6/site-packages/evo/main_ape.py kitti gt_poses.txt cvo_track_poses.txt -va 

#echo "after BA ate:"
#python3 ~/.local/lib/python3.6/site-packages/evo/main_ape.py kitti gt_poses.txt traj_out.txt -va 
#python3 evaluate_ate_scale/evaluate_ate_scale.py --plot after_traj.png --scale 1 --verbose gt_poses.txt traj_out.txt

#python3 ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti cvo_track_poses.txt traj_out.txt --ref  gt_poses.txt -p --plot_mode xz  --align 


