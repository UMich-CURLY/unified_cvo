export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. 

date=$1
clear


for difficulty in Easy #Hard
do

    echo "new seq $i"
    skylabel=(196 112 -- 130  196 146 130)
    seqs=(abandonedfactory gascola hospital seasidetown seasonsforest seasonsforest_winter soulcity)
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
    
    
      ./build/bin/cvo_align_gpu_rgbd_tartan $dataset_folder cvo_params/cvo_outdoor_params.yaml \
                                            tartan_rgbd_${i}_${date}.txt 0 30000  $sky

      
      mv tartan_rgbd_${i}_${date}.txt  $folder/${seq}.txt
      cp $dataset_folder/poses.txt $folder/gt.txt

#      python scripts/xyzq2kitti.py cvo_tartan_outdoor_$i.txt results/tartan/outdoor_easy_kitti_$i.txt 
      #python ~/.local/lib/python3.10/site-packages/evo/main_traj.py kitti --ref ground_truth/tartan/outdoor_easy_kitti_$i.txt  results/tartan/outdoor_easy_kitti_$i.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot -p
      done
done
