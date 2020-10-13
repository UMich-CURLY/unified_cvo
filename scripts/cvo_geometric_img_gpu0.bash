
cd build && make -j && cd .. && \
#for i in  02
for i in 04 05 01 10 06 07 08 02 00 03  09  
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_geometric_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/$i cvo_params/cvo_geometric_params_img_gpu0.yaml \
                                       cvo_geometric_img_gpu0_$i.txt 0 100000
 	
done
