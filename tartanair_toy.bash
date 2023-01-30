cd build
make -j
cd ..
for init_angle in 10
do
	for ratio in 0.0 0.1
	do
		
#gdb -ex run --args \
    ./build/bin/cvo_irls_rand_tartanairpcd ./demo_data/ cvo_params/cvo_intensity_params_irls_exp_tartanair.yaml 4 ${init_angle} 1 tartanair_toy_exp_${init_angle}_${ratio} 1 $ratio 0.01 0.5 0  190 210 230 250	#cat toy_exp_${init_angle}_${ratio}/cvo_err_bunny.txt
		sleep 2

	
	done
done