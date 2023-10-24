#/bin/bash
#seq=$1
date=$1
frame=$2

seq=freiburg3_structure_texture_near

cd build 
make -j 
cd ..
#gdb -ex run --args \
       	./build/bin/cvo_irls_tum  /home/rayzhang/media/Samsung_T5/tum/${seq} cvo_params/cvo_intensity_params_irls_tum.yaml /home/`whoami`/slam_eval/tum_${seq}_${date}/${frame}_graph.txt 1 /home/rayzhang/media/Samsung_T5/tum/${seq}/ 
#gdb -ex run --args \
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq} cvo_params/cvo_intensity_params_irls_tum.yaml /home/rayzhang/dsm/tum_fr1_desk.backup/${frame}_graph.txt 1 /home/rayzhang/dsm/tum_fr1_desk.backup/  #covisMap${frame}.pcd 
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/eth3d/${seq} cvo_params/cvo_intensity_params_irls_tum.yaml /home/rayzhang/dsm/eth3d_ceiling_1/${frame}_graph.txt 1 /home/rayzhang/dsm/eth3d_ceiling_1/  #covisMap${frame}.pcd 
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq} cvo_params/cvo_intensity_params_irls_tum.yaml /home/rayzhang/dsm/sunny_results/tum_freiburg1_${seq}${date}/${frame}_graph.txt 1 /home/rayzhang/dsm/sunny_results/tum_freiburg1_${seq}${date}/  #covisMap${frame}.pcd 
echo "Evaluation: "
echo "before BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py --plot before_traj.png --scale 1  --verbose /home/rayzhang/media/Samsung_T5/tum/${seq}/groundtruth.txt cvo_track_poses.txt 
echo "after BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py --plot after_traj.png --scale 1 --verbose /home/rayzhang/media/Samsung_T5/tum/${seq}/groundtruth.txt traj_out.txt


