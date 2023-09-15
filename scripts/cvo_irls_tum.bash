#/bin/bash
seq=$1
date=_${2}
frame=$3

cd build 
make -j 
cd ..
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg3_structure_notexture_near cvo_params/cvo_intensity_params_irls_tum.yaml graph_defs/fr3/tum_fr3_structure_notexture_near/10_graph.txt /home/rayzhang/media/Samsung_T5/tum/freiburg3_structure_notexture_near/CVO.txt  
#gdb -ex run --args \
./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq} cvo_params/cvo_intensity_params_irls_tum.yaml /home/rayzhang/dsm/tum_fr1_desk.backup/${frame}_graph.txt 1 /home/rayzhang/dsm/tum_fr1_desk.backup/  #covisMap${frame}.pcd 
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq} cvo_params/cvo_intensity_params_irls_tum.yaml /home/rayzhang/dsm/sunny_results/tum_freiburg1_${seq}${date}/${frame}_graph.txt 1 /home/rayzhang/dsm/sunny_results/tum_freiburg1_${seq}${date}/  #covisMap${frame}.pcd 
echo "Evaluation: "
echo "before BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py --plot before_traj.png --scale 1  --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq}/groundtruth.txt cvo_track_poses.txt 
echo "after BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py --plot after_traj.png --scale 1 --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_${seq}/groundtruth.txt traj_out.txt


