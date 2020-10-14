export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
#for i in 01
for i in 01  04 07 10 05 02 00 03 06 08 09 
	
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_semantic_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/$i cvo_params/cvo_semantic_params_img_gpu0.yaml \
                                       cvo_img_semantic_gpu0_$i.txt 0 100000

 	
done
