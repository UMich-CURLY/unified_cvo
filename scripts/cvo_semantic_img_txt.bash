export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
for i in  05 
#for i in 04 05 10 00 01 02 03 06 07 08 09 
do
	echo "new seq $i"
	#gdb -ex=r --args \
    ./build/bin/cvo_align_gpu_semantic_txt /home/v9999/media/seagate_2t/kitti/stereo/$i cvo_params/cvo_semantic_params_img.yaml \
                                       cvo_img_semantic_$i.txt 0 2  $i
 	
done
