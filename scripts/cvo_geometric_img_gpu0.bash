
cd build && make -j && cd .. && \
#for i in  01
for i in 04 05 01 10 09  08 02 00 06 07 03   
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_geometric_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/$i cvo_params/cvo_geometric_params_img_gpu0.yaml \
                                       cvo_geometric_img_gpu0_$i.txt 0 10000
 	
done
