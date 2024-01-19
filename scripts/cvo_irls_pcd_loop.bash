
cd build
cmake ..
make -j
cd ..
cd build_debug
cmake ..
make -j
cd ..

export CUDA_VISIBLE_DEVICES=0

dataset_folder=$1

date=$2
clear

echo " Current Seq: ${i}"        
folder=${dataset_folder}_${date}
rm -rf $folder
mkdir -p $folder



data_dir=$dataset_folder/pcd
lc_file=${dataset_folder}/loop_pairs.txt
init_traj_file=${dataset_folder}/cassie_pose.txt
rm *.pcd
mkdir -p ${folder}/pcds
cp $init_traj_file $folder/tracking_full.txt

last_index=`cat ${init_traj_file} | wc -l`

#gdb -ex run --args \
  ./build/bin/cvo_irls_pcd_loop pcd $data_dir cvo_params/cvo_irls_pcd_params.yaml 2 $folder/tracking_full.txt $lc_file  ba.txt 0 0 $last_index 1.0 0.05  5 1 1 0 1 1 0 #> log_kitti_loop_${i}.txt
	        
	        
  mv [0-9]*.pcd ${folder}/pcds/
  mv *.pcd $folder/
  mv pgo.txt pgo.g2o global.txt loop_closures.g2o tracking.txt ba.txt pose_iter*.txt err_wrt_*.txt log_kitti*.txt $folder/

  echo "Result folder: ${folder}"
  #cp ${dataset_folder}/poses.txt $folder/
	        
	        

                # convert traj to kitti format
                #python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt #--is-change-of-basis
                #python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt #--is-change-of-basis
                #python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt #--is-change-of-basis
                #python3 scripts/xyzq2kitti.py ${folder}/pgo.txt  ${folder}/pgo_kitti.txt# --is-change-of-basis
                #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

                #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder




