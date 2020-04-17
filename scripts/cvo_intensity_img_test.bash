
cd build && make -j && cd .. && \
export CUDA_VISIBLE_DEVICES=1
#for i in 07
for i in 06 07 
#for i in 08 09 
do
    ./build/bin/cvo_align_gpu_raw_img /home/v9999/media/seagate_2t_2/kitti/$i/ cvo_params/cvo_intensity_params_img.yaml \
                                       cvo_img_intensity_$i.txt 0 10000
 
done
