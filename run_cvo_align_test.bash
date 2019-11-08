#!/bin/sh

mode=0  # 0 for online generated points 1 for reading txt
path="/home/rayzhang/seagate_2t/kitti/05"
pcd_path="cvo_points/"
calib_name="cvo_calib.txt"
output_file="cvo_kf_tracking.txt"
start_frame=0
inner_product_threshold=0.5

./build/bin/cvo_align $mode $path $pcd_path $calib_name $output_file $start_frame $inner_product_threshold
