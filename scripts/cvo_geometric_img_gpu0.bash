export CUDA_VISIBLE_DEVICES=0
cd build && make -j && cd .. && \
#for i in  10 
#for i in 04 05 01 10 09  08 02 00 06 07 03
for i in 01 10 07 02 05  04 06	
do
	echo "new seq $i"
    ./build/bin/cvo_align_gpu_geometric_img /home/v9999/media/seagate_2t/kitti/stereo/$i cvo_params/cvo_geometric_params_img_gpu0.yaml \
                                       cvo_geometric_img_gpu0_$i.txt 0 20000
 	
done
