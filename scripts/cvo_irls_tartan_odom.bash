cd build
make -j
cd ..

date=$1
clear

for difficulty in Easy #Hard
do
    skylabel=(196 112 -- 130  196 146 130)
    seqs=(abandonedfactory gascola hospital seasidetown seasonsforest seasonsforest_winter soulcity)
    #seqs=(abandonedfactory)

    for ind in ${!seqs[@]}
    do
        i=${seqs[ind]}
        sky=${skylabel[ind]}
	folder=tartan_rgbd_${difficulty}_${i}_${date}
        dataset_folder=/home/rayzhang/media/Samsung_T5/tartanair/$i/${difficulty}/P001/
        echo " Current Seq: ${i} ${difficulty} with sky label ${sky}"        
	rm -rf $folder
	mkdir -p $folder
	rm *.pcd

#                gdb -ex run --args \
        ./build/bin/cvo_irls_tartan_odom $dataset_folder cvo_params/cvo_outdoor_params.yaml cvo_calib_deep_depth.txt 4 tracking.txt ba.txt 0 100000 $sky # > log_tartan_rgbd_${difficulty}_${i}.txt
        mv *.pcd $folder/
        mv tracking.txt ba.txt err_wrt_iters_*.txt groundtruth.txt $folder/
        cp ${dataset_folder}/pose_left.txt $folder/

        # convert traj to kitti format
        python3 scripts/xyzq2kitti.py ${folder}/groundtruth.txt  ${folder}/groundtruth_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/tracking.txt  ${folder}/tracking_kitti.txt
        python3 scripts/xyzq2kitti.py ${folder}/ba.txt  ${folder}/ba_kitti.txt
        #python3 /home/rayzhang/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref ${folder}/groundtruth_kitti.txt ${folder}/tracking_kitti.txt   ${folder}/ba_kitti.txt  -p --plot_mode xyz

        #mv log_tartan_rgbd_${difficulty}_${i}.txt $folder
        sleep 1

    done
done

