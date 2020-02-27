

cd build && make -j && cd ..
export CUDA_VISIBLE_DEVICES=1
for i in 02 03 04 05 
do
    echo "running $i..." && \
        ./build/bin/acvo_align_semantic_gpu_raw_img /home/rzh/media/sda1/ray/datasets/kitti/sequences/${i} cvo_params/acvo_params_gpu.txt accum${i}_semantic.txt 0 10000 && \
        echo "finsh $i.... next"
        
done
