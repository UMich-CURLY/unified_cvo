cd build
make -j
cd ..
for init_angle in 12.5 25 37.5 50
do
	for ratio in 0.0 0.125 0.25 0.375 0.5
	do
		
#gdb -ex run --args \
    ./build/bin/cvo_align_align_run_tartan tartanair_full_semantic/tartanair_toy_exp_${init_angle}_${ratio} cvo_params/cvo_intensity_params_irls_exp_tartanair_semantics.yaml 4 ${init_angle} 40 tartanair_full_semantic/tartanair_toy_exp_${init_angle}_${ratio} 1 $ratio 0.01 0.5 0	#cat toy_exp_${init_angle}_${ratio}/cvo_err_bunny.txt
	sleep 2

	
	done
done