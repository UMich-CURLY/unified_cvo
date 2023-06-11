cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..
#export CUDA_VISIBLE_DEVICES=1
date=$1
clear

    #skylabel=(196 112 -- 130  196 146 130)
    #seqs=( 05 00 08 02 06 09 )
    #seqs=( 05 00 02 06 09 07  )
    seqs=( 05 02  )
    for ind in ${!seqs[@]}
    do
        echo " Current Seq: ${ind}"        
        i=${seqs[ind]}
        #sky=${skylabel[ind]}
	folder=kitti_lidar_${i}_${date}
        dataset_folder=/home/rayzhang/media/Samsung_T5/kitti_lidar/dataset/sequences/${i}/
	#dataset_folder=/home/rzh//media/sdg1/rzh/kitti_lidar/kitti_lidar/dataset/sequences/${i}/
        #lc_file=/home/rzh/unified_cvo/demo_data/kitti_loop_closure/kitti_${i}.txt
        lc_file=/home/rayzhang/unified_cvo/demo_data/kitti_loop_closure/kitti_${i}_loop_closure.g2o

	rm -rf $folder
	mkdir -p $folder
	#cp results/cvo_geometric_img_gpu0_mar21/${i}.txt $folder/tracking_full.txt        
	cp cvo_align_lidar_jun05/${i}.txt $folder/tracking_full.txt        
	rm *.pcd

        ### run global BA
        #gdb -ex run --args \
        #./build/bin/cvo_irls_lidar_loop $dataset_folder cvo_params/cvo_irls_kitti_ba_params.yaml 2 $folder/tracking_full.txt $lc_file  ba.txt 0 0 1000000 2.0 0 0 0 # > log_tartan_rgbd_${difficulty}_${i}.txt
        #gdb -ex run --args \
        ./build/bin/cvo_irls_lidar_loop $dataset_folder cvo_params/cvo_irls_kitti_ba_params.yaml 2 $folder/tracking_full.txt $lc_file  ba.txt 0 0 1000000 1.0 0.1  0 0 1  > log_kitti_loop_${i}.txt
	
        mv *.pcd $folder/
        mv pgo.txt global.txt loop_closures.g2o tracking.txt ba.txt err_wrt_*.txt log_kitti*.txt groundtruth.txt $folder/
        cp ${dataset_folder}/poses.txt $folder/

        # convert traj to kitti format
        python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/pgo.txt  ${folder}/pgo_kitti.txt
        #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

        #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
        sleep 3

    done

