cd build
make -j
cd ..

date=$1
clear

    #skylabel=(196 112 -- 130  196 146 130)
    seqs=( 03 )
    for ind in ${!seqs[@]}
    do
        i=${seqs[ind]}
        #sky=${skylabel[ind]}
	folder=kitti_lidar_${i}_${date}
        dataset_folder=/home/rayzhang/media/Samsung_T5/kitti_lidar/dataset/sequences/$i/
        echo " Current Seq: ${i}"        
	rm -rf $folder
	mkdir -p $folder
	rm *.pcd

        #gdb -ex run --args \
        ./build/bin/cvo_irls_lidar_ba $dataset_folder cvo_params/cvo_irls_kitti_ba_params.yaml 4 tracking.txt ba.txt 0 720 725 # > log_tartan_rgbd_${difficulty}_${i}.txt
        mv *.pcd $folder/
        mv tracking.txt ba.txt err_wrt_iters_*.txt groundtruth.txt $folder/
        cp ${dataset_folder}/poses.txt $folder/

        # convert traj to kitti format
        python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt
        #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

        #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
        sleep 3

    done

