cd build
make -j
cd ..
date=$1
#for i in abandonedfactory endofworld gascola soulcity abandonedfactory_night ocean seasidetown #abandonedfactory abandonedfactory_night seasonsforest sea
#for i in hospital #westerndesert  #hospital  #abandonedfactory_night #neighborhood  #seasidetown abandonedfactory  seasonsforest gascola hospital
#for i in eth3d_sfm_lab_room_1 eth3d_sfm_garden sfm_house_loop eth3d_repetitive eth3d_ceiling_1 eth3d_planar_1 #westerndesert  #hospital  #abandonedfactory_night #neighborhood  #seasidetown abandonedfactory  seasonsforest gascola hospital
#for i in  eth3d_ceiling_1 eth3d_planar_1 #westerndesert  #hospital  #abandonedfactory_night #neighborhood  #seasidetown abandonedfactory  seasonsforest gascola hospital
for i in tum_freiburg1_desk2 #tum_freiburg1_desk2 tum_freiburg1_room
do
    
	#./build/bin/stack_pcd_viewer /home/rayzhang/kitti_color_05 2700 0 0
   ./build/bin/stack_pcd_viewer /home/rayzhang/dsm/sunny_results/${i}_$date 2000 180 0
   #cat /home/rayzhang/dsm/$i/${i}_graph.txt
   #./build/bin/stack_pcd_viewer /home/rayzhang/dsm/ 0 0 0
done

