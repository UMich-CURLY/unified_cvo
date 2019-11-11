mode=0  # 0 for online generated points 1 for reading txt
# dataset="05"

for dataset in 04
do
    path="/media/justin/LaCie/data/kitti/sequences/"$dataset
    pcd_path="cvo_points/"
    calib_name="cvo_calib.txt"
    output_file="results/cvo_f2f_tracking_relative_"$dataset"_semantic.txt"
    start_frame=0
    num_classes=19

    ./build/bin/cvo_f2f $mode $path $pcd_path $calib_name $output_file $start_frame $dataset $num_classes
done