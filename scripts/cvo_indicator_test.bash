export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
# for i in freiburg1_desk
for i in 05
#for i in 00 01 02 04 05 06 07 08 09 10 #01 02 03 04 05 06 07 08 09 10
# for i in freiburg1_desk freiburg1_desk2 freiburg1_room freiburg1_360 freiburg1_teddy freiburg1_xyz freiburg1_rpy freiburg1_plant freiburg1_floor freiburg3_nostructure_texture_far freiburg3_nostructure_texture_near freiburg3_structure_notexture_far freiburg3_structure_notexture_near freiburg3_structure_texture_far freiburg3_structure_texture_near freiburg3_nostructure_notexture_far freiburg3_nostructure_notexture_near
do
    echo ""
    echo "/********************** Sequence "$i" *************************/"
    for method in geometric_cvo semantic_cvo #fast_gicp #results_semantic_cvo #results_gicp_new results_ndt ground_truth results_cvo_geometric_stereo #results_gicp results_color_cvo 
    do

    # gdb -ex run --args \
     # ./build/bin/cvo_align_gpu_lidar_raw /home/cel/data/kitti/sequences/$i cvo_params/cvo_geometric_params_gpu.yaml \
        #                                results/lidar_geometric_result/cvo_geometric_$i"_indicator_evaluation_afterpointselection.txt" 0 2

        # mv inner_product_history.txt indicator_evaluation/$i"_afterpointselection_init_guess_inner_product_history.txt"
        # mv function_angle_history.txt indicator_evaluation/$i"_afterpointselection_init_guess_function_angle_history.txt"

    #  ./build/bin/cvo_align_gpu_raw_img /home/cel/data/kitti/sequences/$i cvo_params/cvo_intensity_params_img.yaml \
                                    #    cvo_img_intensity_$i"_test_indicator.txt" $frame 1 #$angle #100000
        # mv stereo_semi_dense_temp.pcd indicator_evaluation/stereo_semi_dense/$i"_"$frame"_stereo_semi_dense.pcd"
        # mv stereo_full_temp.pcd /home/cel/data/kitti/sequences/$i/stereo_full_pcd/0000$frame".pcd"

                                       #cvo_align_gpu_lidar_loam

        # ./build/bin/cvo_evaluate_indicator cvo_params/cvo_intensity_params_img.yaml \
        #                                     indicator_evaluation/$i"_evaluate_indicator_full.txt" \
        #                                     $i

        # ./build/bin/cvo_evaluate_indicator /home/cel/outdoor_cvo_rgbd/cvo_params/cvo_rgbd_params.yaml \
        #                                     indicator_evaluation/$i"_evaluate_indicator_full.txt" \
        #                                     $i

        #./build/bin/cvo_indicator_in_sequence /home/v9999/media/seagate_2t/kitti/stereo/$i \
        #                                        cvo_params/cvo_geometric_params_img_gpu0.yaml \
        #                                        $i \
        #                                        $method \
        #                                        baselines/stereo/$method/$i.txt
                                       
                                      
        #./build/bin/cvo_indicator_in_sequence /home/v9999/media/seagate_2t/kitti/stereo/$i \
        #                                        cvo_params/cvo_geometric_params_img_gpu0.yaml \
        #                                        $i \
        #                                        $method \
        #                                        results/cvo_geometric_img_gpu0_oct23/$i.txt
        
	./build/bin/cvo_indicator_in_sequence  /home/rayzhang/media/Samsung_T5/kitti_stereo/dataset/sequences/kitti/$i \
                                                cvo_params/cvo_intensity_params_img_gpu0.yaml \
                                                $i \
                                                $method \
                                                results/cvo_intensity_img_gpu0_oct25_best/$i.txt
     done 
done
