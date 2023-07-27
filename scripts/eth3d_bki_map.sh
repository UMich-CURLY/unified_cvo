cd build
make -j
cd ..
gdb -ex run --args \
./build/bin/eth3d_local_mapping /home/rayzhang/media/Samsung_T5/eth3d/sfm_lab_room_1 cvo_params/cvo_intensity_params_irls_eth3d.yaml 0 50 /home/rayzhang/media/Samsung_T5/eth3d/sfm_lab_room_1/CVO.txt
