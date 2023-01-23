export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd ..
#for i in  05
#for i in  01 07 10  00 02 03 04 05 06 08 09
#for i in carwelding hospital
#do
#	echo "new seq $i"
      #./build/bin/cvo_align_gpu_rgbd_tartan /home/rayzhang/media/Samsung_T5/tartanair/$i/Easy/P001/ cvo_params/cvo_indoor_params.yaml \
      #                                 cvo_tartan_indoor_$i.txt 0 30000
 #     python scripts/xyzq2kitti.py cvo_tartan_indoor_$i.txt results/tartan/indoor_easy_kitti_$i.txt
#	python ~/.local/lib/python3.10/site-packages/evo/main_traj.py kitti --ref ground_truth/tartan/indoor_easy_kitti_$i.txt  results/tartan/indoor_easy_kitti_$i.txt  --plot_mode xyz --align --save_plot results/tartan/${i}_plot -p

#done

for i in oldtown
do
	echo "new seq $i"
      ./build/bin/cvo_align_gpu_stereo_tartan /media/sdc1/rzh/tartanair/$i/Easy/P001 cvo_params/cvo_outdoor_params.yaml \
                                       cvo_tartan_outdoor_$i.txt 0 30000
#      python scripts/xyzq2kitti.py cvo_tartan_outdoor_$i.txt results/tartan/outdoor_easy_kitti_$i.txt
	#python ~/.local/lib/python3.10/site-packages/evo/main_traj.py kitti --ref ground_truth/tartan/outdoor_easy_kitti_$i.txt  results/tartan/outdoor_easy_kitti_$i.txt   --plot_mode xyz --align --save_plot results/tartan/${i}_plot -p
done
