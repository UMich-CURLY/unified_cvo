export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
#for i in  05
#for i in  01 07 10  00 02 03 04 05 06 08 09 
for i in 04 01 03 10 07  05  02 06	
#for i in  05  04 06	
do
	echo "new seq $i"
      #./build/bin/cvo_align_gpu_img /home/rayzhang/media/Samsung_T5/kitti/$i cvo_params/cvo_intensity_params_img_gpu0.yaml \
#	gdb -ex run --args \
      ./build/bin/cvo_align_gpu_img /home/rayzhang/media/Samsung_T5/kitti_stereo/dataset/sequences/kitti/$i cvo_params/cvo_intensity_params_img_gpu0.yaml \
                                       cvo_intensity_img_jan9_$i.txt 0 100000000

 	
done
