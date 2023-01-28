cd build
make -j
cd ..
for init_angle in 10 20 30 40 50 
do
	for ratio in 0.0 0.1 0.2 0.3 0.4 0.5
	do
		
#gdb -ex run --args \
    ./build/bin/cvo_irls_rand_pcd demo_data/example_ cvo_params/cvo_intensity_params_irls_bunny.yaml 4 ${init_angle} 1 toy_exp_${init_angle}_${ratio} 1 $ratio 0.01 0.5 0	#cat toy_exp_${init_angle}_${ratio}/cvo_err_bunny.txt
		sleep 2
	done
done

