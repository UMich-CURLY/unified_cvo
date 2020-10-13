
cd build && make -j && cd .. && \
#for i in  02
for i in 03 04 05 06 07 08 00 01 02 09 10 
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_geometric_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/$i cvo_params/cvo_intensity_params_img_gpu0.yaml \
                                       cvo_intensity_img_gpu0_$i.txt 0 10000
 	
done
