mode=0  # 0 for online generated points 1 for reading txt
path="/media/justin/LaCie/data/kitti/sequences/05"
pcd_path="cvo_points/"
calib_name="cvo_calib.txt"
output_file="cvo_kf_tracking.txt"
start_frame=0
inner_product_threshold=0.25

./build/bin/cvo_align $mode $path $pcd_path $calib_name $output_file $start_frame $inner_product_threshold


mode=0  # 0 for online generated points 1 for reading txt
# dataset="05"

for dataset in 01 04 06
do
    path="/media/justin/LaCie/data/kitti/sequences/"$dataset
    pcd_path="cvo_points/"
    calib_name="cvo_calib.txt"
    output_file="cvo_f2f_tracking_relative_"$dataset".txt"
    start_frame=0
    inner_product_threshold=0.5

    ./build/bin/cvo_f2f $mode $path $pcd_path $calib_name $output_file $start_frame $inner_product_threshold $dataset
done