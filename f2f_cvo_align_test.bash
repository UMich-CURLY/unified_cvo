mode=0  # 0 for online generated points 1 for reading txt

for dataset in 05
do
    path="/home/cel/PERL/datasets/kitti_dataset/sequences/"$dataset
    pcd_path="cvo_points/"
    calib_name="cvo_calib.txt"
    output_file="results/cvo_f2f_tracking_relative_lidar_"$dataset".txt"
    start_frame=0
    num_classes=19

    ./build/bin/cvo_f2f $mode $path $pcd_path $calib_name $output_file $start_frame $dataset #$num_classes
done

# ./build/bin/cvo_f2f 0 /home/cel/PERL/datasets/kitti_dataset/sequences/05 cvo_points/ cvo_calib.txt results/cvo_f2f_tracking_relative_05.txt 0 05 19
