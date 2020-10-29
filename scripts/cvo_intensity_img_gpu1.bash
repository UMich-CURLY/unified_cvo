
cd build && make -j && cd .. && \
for i in  03
#for i in 03 04 05 06 07 08 00 01 02 09 10
#for i in 10 01  02 05  04 06 03
#for i in 09 07 00 08 
	
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_intensity_img /home/v9999/media/seagate_2t/kitti/stereo/$i cvo_params/cvo_intensity_params_img_gpu0.yaml \
                                       cvo_intensity_img_gpu0_oct25_$i.txt 0 200000
 	
done
