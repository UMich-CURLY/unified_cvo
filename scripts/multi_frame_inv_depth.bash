 cd build_debug
make -j
cd ..
 cd build
make -j
cd ..

#for seq in abandonedfactory  #hospital #soulcity #hospital #abandonedfactory_night0
#do
#gdb -ex run --args 
#	./build/bin/cvo_inv_depth_tartan /home/rayzhang/media/Samsung_T5/tartanair/$seq/Easy/P001/ 0 /home/rayzhang/slam_eval/tartan_rgbd_Easy_${seq}_jan31/${1}_graph.txt.10    /home/rayzhang/media/Samsung_T5/tartanair/$seq/Easy/P001/pose_left.txt   1 1
#done

####### tum
#gdb -ex run --args \
	./build_debug/bin/cvo_inv_depth_tartan /home/rayzhang/media/Samsung_T5/tum/freiburg1_desk/ 2 ~/media/Samsung_T5/tum/freiburg1_desk/${1}_graph.txt  /run/media/rayzhang/Samsung_T5/tum/freiburg1_desk/groundtruth.txt   1 1


####### kitti
#    gdb -ex run --args \
#	./build/bin/cvo_inv_depth_tartan /home/rayzhang/media/Samsung_T5/kitti_stereo/dataset/sequences/05/ 1 /home/rayzhang/dsm/kitti_color_05/${1}_graph.txt  ground_truth/05.txt   100 1
#	./build/bin/cvo_inv_depth_tartan /home/rayzhang/media/Samsung_T5/kitti_stereo/dataset/sequences/05/ 1  /home/rayzhang/media/Samsung_T5/kitti_stereo/dataset/sequences/05/${1}_graph.txt ground_truth/05.txt   1 1
