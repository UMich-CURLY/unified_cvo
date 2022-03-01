export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. 
#for i in  05
#for i in  01 07 10  00 02 03 04 05 06 08 09
#start_ind=112
#end_ind=122
#for i in hospital 	
#do 
#	echo "new seq $i"
#      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_indoor_params.yaml \
#                                       cvo_tartan_indoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt
 #     python scripts/xyzq2kitti.py cvo_tartan_indoor_$i.txt results/tartan/indoor_easy_kitti_$i.txt 
#	python ~/.local/lib/python3.10/site-packages/evo/main_traj.py kitti --ref ground_truth/tartan/indoor_easy_kitti_$i.txt  results/tartan/indoor_easy_kitti_$i.txt  --plot_mode xyz --align --save_plot results/tartan/${i}_plot -p
#done

#start_ind=563
#end_ind=760
#for i in carwelding
#do 
#	echo "new seq $i"
#      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_indoor_params.yaml \
#                                       cvo_tartan_indoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt
 #     python scripts/xyzq2kitti.py cvo_tartan_indoor_$i.txt results/tartan/indoor_easy_kitti_$i.txt 
#	python ~/.local/lib/python3.10/site-packages/evo/main_traj.py kitti --ref ground_truth/tartan/indoor_easy_kitti_$i.txt  results/tartan/indoor_easy_kitti_$i.txt  --plot_mode xyz --align --save_plot results/tartan/${i}_plot -p
#done

#for i in abandonedfactory_night abandonedfactory seasonsforest seasonsforest_winter	
#do
#	echo "new seq $i"
#      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
#                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
                                       #cvo_tartan_outdoor_${i}.txt ${start_ind} ${end_ind} 
#      python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
#      python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
#done


start_ind=400
end_ind=450
for i in abandonedfactory #seasonsforest seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt 
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done

start_ind=188
end_ind=200
for i in abandonedfactory #seasonsforest seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt 
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=190
end_ind=210
for i in abandonedfactory_night #seasonsforest seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt 
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=30
end_ind=45
for i in abandonedfactory_night #seasonsforest seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt 
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=460
end_ind=480
for i in abandonedfactory_night #seasonsforest seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt 
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=780
end_ind=800
for i in abandonedfactory_night #seasonsforest seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind} > tartan_log_${i}_${start_ind}_${end_ind}.txt 
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=60
end_ind=120
for i in seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=750
end_ind=800
for i in seasonsforest_winter	
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done



start_ind=210
end_ind=220
for i in seasonsforest
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done




start_ind=0
end_ind=80
for i in endofworld
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done



start_ind=150
end_ind=250
for i in seasidetown
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done


start_ind=40
end_ind=130
for i in ocean
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python scripts/xyzq2kitti.py cvo_tartan_outdoor_${i}_${start_ind}_${end_ind}.txt results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt
      #python scripts/xyzq2kitti.py /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/pose_left.txt results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt ${start_ind} ${end_ind}
      #python ~/.local/lib/python3.6/site-packages/evo/main_traj.py kitti --ref results/tartan/gt_outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt  results/tartan/outdoor_easy_kitti_${i}_${start_ind}_${end_ind}.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot 
done
