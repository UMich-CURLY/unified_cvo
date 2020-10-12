
cd build && make -j && cd .. && \
#for i in  04 05
for i in 03 04 05 06 07 08 00 01 02 09 10 
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_geometric_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/$i cvo_params/cvo_geometric_params_img.yaml \
                                       cvo_geometric_img_$i.txt 0 4
 	
done
