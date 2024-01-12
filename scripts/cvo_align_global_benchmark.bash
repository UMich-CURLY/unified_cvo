
clear

date=$1

cd build
make -j
cd ..

for angle in 180 #90
do
for outlier_on in  0 #1
do
    for outlier_ratio in 1.0 #0.8 #0.8
    do
	    for crop_ratio in  0.25 0.375 0.5
	    do  
		 #   gdb -ex run --args \
        ./build/bin/cvo_align_gpu_benchmark_two_pcd_global demo_data/bunny.pcd cvo_params/cvo_global_params.yaml 0.5 $angle 40 result_global_${angle}_${outlier_ratio}_${crop_ratio}_${date} $outlier_on $outlier_ratio 0.01 0.1 0.03 0.05 $crop_ratio       
done    
done

done
done
