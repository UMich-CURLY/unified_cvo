#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
// #include <opencv2/opencv.hpp>
#include "dataset_handler/EthzHandler.hpp"
#include "graph_optimizer/Frame.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;

int main(int argc, char *argv[])
{
    // load dataset
    std::string dataset_folder = "/home/root/data/ethz/";
    cvo::EthzHandler dataset_local(dataset_folder, cvo::EthzHandler::FrameType::LOCAL);
    cvo::EthzHandler dataset_global(dataset_folder, cvo::EthzHandler::FrameType::GLOBAL);

    // init viewer
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");

    // stack the global point cloud

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_stacked(new pcl::PointCloud<pcl::PointXYZI>);
    int n = 3; // dataset_global.get_total_number()
    for (int i = 0; i < n; i++)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        dataset_global.read_next_lidar(pc);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb(pc, rand() % 255, rand() % 255, rand() % 255);
        viewer.addPointCloud<pcl::PointXYZI>(pc, rgb, "frame" + to_string(i) + "_global");
        dataset_global.next_frame_index();

        // stack the pc
        *pc_stacked += *pc;
    }
    // viewer.spin();

    // save the pc_stacked
    // pcl::io::savePCDFileASCII(dataset_folder + "pc_stacked.pcd", *pc_stacked);

    // clear the viewer and add the stacked pc
    viewer.removeAllPointClouds();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb(pc_stacked, 255, 255, 255);
    viewer.addPointCloud<pcl::PointXYZI>(pc_stacked, rgb, "pc_stacked");
    // viewer.spin();

    // read gt
    viewer.removeAllPointClouds();
    std::vector<double> timestamp;
    std::vector<Eigen::Matrix4d> gt_poses;
    dataset_local.read_ground_truth_poses(timestamp, gt_poses);
    // print dataset_global.get_total_number();
    std::cout << "dataset size = " << dataset_global.get_total_number() << std::endl;
    std::cout << "gt poses size = " << gt_poses.size() << std::endl;
    n = 5; // dataset_global.get_total_number();
    for (int i = 0; i < n; i++)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
        dataset_local.read_next_lidar(pc);
        // transform the pc according to gt poses
        Eigen::Matrix4d T = gt_poses[i];
        pcl::transformPointCloud(*pc, *pc, T);
        // viz
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb(pc, rand() % 255, rand() % 255, rand() % 255);
        viewer.addPointCloud<pcl::PointXYZI>(pc, rgb, "frame" + to_string(i) + "_global");
        dataset_local.next_frame_index();
    }
    viewer.spin();

    return 0;
}
