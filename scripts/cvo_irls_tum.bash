#/bin/bash

cd build 
make -j 
cd ..
./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2 cvo_params/cvo_intensity_params_irls_tum.yaml graph_defs/graph_tum2.txt /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2/CVO.txt 
echo "Evaluation: "
echo "before BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py  --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2/groundtruth.txt cvo_track_poses.txt 
echo "after BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py  --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2/groundtruth.txt traj_out.txt


