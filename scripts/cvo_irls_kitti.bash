#/bin/bash

cd build 
make -j 
cd ..
gdb -ex run --args \
./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2 cvo_params/cvo_intensity_params_irls_tum.yaml graph_defs/tum/3_graph.txt /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2/CVO.txt 
echo "Evaluation: "
echo "before BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py --plot before_traj.png --scale 1  --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2/groundtruth.txt cvo_track_poses.txt 
echo "after BA ate:"
python3 evaluate_ate_scale/evaluate_ate_scale.py --plot after_traj.png --scale 1 --verbose /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk2/groundtruth.txt traj_out.txt


