mode=0  # 0 for online generated points 1 for reading txt
path="/media/justin/LaCie/data/kitti/sequences/05"
pcd_path="cvo_points/"
calib_name="camera.txt"
output_file="cvo_f2f_tracking_relative.txt"
start_frame=0
inner_product_threshold=0.5

./build/bin/cvo_f2f $mode $path $pcd_path $calib_name $output_file $start_frame $inner_product_threshold