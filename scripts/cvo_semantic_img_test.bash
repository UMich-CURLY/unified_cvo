export CUDA_VISIBLE_DEVICES=0
#cd build && make -j && cd .. && \
for i in  05 
#for i in 04 05 10 00 01 02 03 06 07 08 09 
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_semantic_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/$i cvo_params/cvo_semantic_params_img.yaml \
                                       cvo_img_semantic_$i.txt 0 50000
 	
done
