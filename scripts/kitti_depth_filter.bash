
seq=$1
frame=$2

direll=$3
normalell=$4

cd build
make -j
cd ..
./build/bin/kitti_depth_filtering  ..//media/Samsung_T5/kitti_stereo/dataset/sequences/kitti/${seq}/ cvo_params/cvo_intensity_params_irls_kitti.yaml results/cvo_intensity_img_gpu0_oct25_best/${seq}.txt $frame 5 $direll $normalell
