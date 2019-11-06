mode=0  # 0 for online generated points 1 for reading txt
path="/media/justin/LaCie/data/kitti/sequences/05"
pcd_path="/media/justin/LaCie/data/kitti/sequences/05/cvo_points/"
calib_name="camera.txt"
output_file="cvo_kf_tracking.txt"
start_frame=1
inner_product_threshold=0.00174

./build/bin/cvo_align $mode $path $pcd_path $calib_name $output_file $start_frame $inner_product_threshold