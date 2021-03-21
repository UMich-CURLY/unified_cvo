export CUDA_VISIBLE_DEVICES=0
cd build && make -j4 && cd .. && \
for i in  07 
#for i in 00 01 02 03 04 05 06 07 08 09 10
#for i in 01 10 07 02 05  04 06	
do
	echo "new seq $i"
        #gdb --ex run --args  \
     #       nvprof  \
    ./build/bin/cvo_align_gpu_img /home/rayzhang/code/docker_home/media/Samsung_T5/kitti/$i cvo_params/cvo_geometric_params_img_gpu0.yaml \
                                       cvo_geometric_img_gpu0_$i.txt 0 200000
 	
done
