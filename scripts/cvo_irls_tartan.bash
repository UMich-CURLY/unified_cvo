#/bin/bash

cd build 
make -j 
cd ..
#gdb -ex run --args \
#./build/bin/cvo_irls_tum /home/rayzhang/media/Samsung_T5/tum/freiburg3_structure_notexture_near cvo_params/cvo_intensity_params_irls_tum.yaml graph_defs/fr3/tum_fr3_structure_notexture_near/10_graph.txt /home/rayzhang/media/Samsung_T5/tum/freiburg3_structure_notexture_near/CVO.txt  
./build/bin/cvo_irls_tartan /home/rayzhang/media/Samsung_T5/tartanair/office/Easy/P001/  cvo_params/cvo_indoor_params.yaml graph_defs/tartan/tartan_easy_office/50_graph.txt 5


