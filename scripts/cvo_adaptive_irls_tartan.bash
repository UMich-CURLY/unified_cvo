
cd build && make -j && cd .. && \

for seq in abandonedfactory  #hospital #soulcity #hospital #abandonedfactory_night
do
#./build/bin/cvo_irls_tartan /home/rayzhang/media/Samsung_T5/tartanair/$seq/Easy/P001/  cvo_params/cvo_outdoor_params.yaml /home/rayzhang/dsm/tartan_Easy_${seq}/${1}_graph.txt 1 /home/rayzhang/dsm/tartan_Easy_${seq}/ covisMap${1}.pcd
#./build/bin/cvo_irls_tartan /home/rayzhang/media/Samsung_T5/tartanair/$seq/Easy/P001/  cvo_params/cvo_outdoor_params.yaml /home/rayzhang/dsm/tartan_Easy_${seq}/${1}_graph.txt 1 /home/rayzhang/dsm/tartan_Easy_${seq}/ covisMap${1}.pcd 

#gdb -ex run --args \
./build/bin/cvo_irls_tartan /home/rayzhang/media/Samsung_T5/tartanair/$seq/Easy/P001/  cvo_params/cvo_outdoor_params.yaml /home/rayzhang/slam_eval/tartan_semantic_Easy_${seq}_jan31/${1}_graph.txt 1 /home/rayzhang/slam_eval/tartan_semantic_Easy_${seq}_jan31/ 0 #covisMap${1}.pcd    

done
