cd build
make -j
cd ..
for init_angle in 0.01 0.1 0.25
do
	for ratio in 0.01 0.1 0.25
	do
#gdb -ex run --args \
    ./build/bin/cvo_irls_rand_pcd demo_data/bunny.pcd cvo_params/cvo_intensity_params_irls_bunny.yaml 4 ${init_angle} 50 toy_exp_${init_angle}_${ratio} 1 $ratio 0.005 0.5

	done
done
