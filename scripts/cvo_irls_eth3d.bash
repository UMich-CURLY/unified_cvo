#/bin/bash
seq=$1
frame=$2


# sfm_lab_room_1:   9   55

cd build 
make -j 
cd ..
# dsm 
#gdb -ex run --args \
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/eth3d/${seq} /home/rayzhang/dsm/Examples/TumExample/cvo_rgbd_params_eth3d.yaml /home/rayzhang/dsm/eth3d_${seq}/${frame}_graph.txt 1 /home/rayzhang/media/Samsung_T5/eth3d/${seq}/

# single tests
./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/eth3d/${seq} /home/rayzhang/dsm/Examples/TumExample/cvo_rgbd_params_eth3d.yaml /home/rayzhang/unified_cvo/${frame}_graph.txt 1 /home/rayzhang/media/Samsung_T5/eth3d/${seq}/

cat /home/rayzhang/dsm/eth3d_${seq}/${frame}_graph.txt 
#echo "Evaluation: "
#echo "before BA ate:"
#python3 evaluate_ate_scale/evaluate_ate_scale.py --plot before_traj.png --scale 1  --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq}/groundtruth.txt cvo_track_poses.txt 
#echo "after BA ate:"
#python3 evaluate_ate_scale/evaluate_ate_scale.py --plot after_traj.png --scale 1 --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq}/groundtruth.txt traj_out.txt


