

#for i in abandonedfactory endofworld gascola soulcity abandonedfactory_night ocean seasidetown #abandonedfactory abandonedfactory_night seasonsforest sea
#for i in hospital #westerndesert  #hospital  #abandonedfactory_night #neighborhood  #seasidetown abandonedfactory  seasonsforest gascola hospital
for i in tum_fr1_desk #westerndesert  #hospital  #abandonedfactory_night #neighborhood  #seasidetown abandonedfactory  seasonsforest gascola hospital
do
    
	#./build/bin/stack_pcd_viewer /home/rayzhang/kitti_color_05 2700 0 0
   ./build/bin/stack_pcd_viewer /home/rayzhang/dsm/$i 270 100 0
   cat /home/rayzhang/dsm/$i/${i}_graph.txt
   #./build/bin/stack_pcd_viewer /home/rayzhang/dsm/ 0 0 0
done

