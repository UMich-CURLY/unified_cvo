
cd build && make -j8 && \
nvprof ./bin/acvo_align_gpu /home/rayzhang/outdoor_cvo/gpu_test_data/cvo_points_05/ \
/home/rayzhang/outdoor_cvo/cvo_params/acvo_semantic_params_gpu.txt \
rela.txt accum.txt 0  2760

