export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
for i in  05
#for i in  01 07 10  00 02 03 04 05 06 08 09 
#for i in 01 10 07 02 05  04 06	
#for i in  05  04 06	
do
	echo "new seq $i"
#	gdb -ex run --args \
      ./build/bin/cvo_align_gpu_img /home/rayzhang/media/Samsung_T5/kitti/$i cvo_params/cvo_intensity_params_img_gpu0.yaml \
                                       cvo_intensity_img_gpu0_$i.txt 0 3

 	
done
