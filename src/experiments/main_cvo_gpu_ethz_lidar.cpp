#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
// #include <opencv2/opencv.hpp>
#include "dataset_handler/EthzHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "utils/LidarPointDownsampler.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "utils/ImageDownsampler.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;

int main(int argc, char *argv[])
{
  // load dataset
  std::string dataset_folder = argv[1];
  cvo::EthzHandler dataset(dataset_folder, cvo::EthzHandler::FrameType::LOCAL);
  int total_iters = dataset.get_total_number();
  string cvo_param_file(argv[2]);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  dataset.set_start_index(start_frame);
  int max_num = std::stoi(argv[5]);

  // init viewer
  std::vector<double> timestamp;
  std::vector<Eigen::Matrix4d> gt_poses;
  dataset.read_ground_truth_poses(timestamp, gt_poses);

  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0.0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  accum_output<<"1 0 0 0 0 1 0 0 0 0 1 0\n"<<std::flush;

  std::cout<<"new cvo_align, total_iters is "<<total_iters<<"\n";
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  
    
  // std::cout << "dataset size = " << dataset.get_total_number() << std::endl;
  // std::cout << "gt poses size = " << gt_poses.size() << std::endl;
  double total_time = 0;
  int n = dataset.get_total_number();
  int i = start_frame;
  for (; i<min(total_iters, start_frame+max_num)-1; i++)
  {

    dataset.set_start_index(i);
    pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
    dataset.read_next_lidar(source_pc);
    std::shared_ptr<cvo::CvoPointCloud> source
      (new cvo::CvoPointCloud(source_pc, 5000, -1,
                                                                     cvo::CvoPointCloud::PointSelectionMethod::FULL));
    std::unordered_set<const cvo::CvoPoint *> selected_pts;
    std::shared_ptr<cvo::CvoPointCloud> source_downsampled = cvo::voxel_downsample(source, 0.1, selected_pts, cvo::CvoPointCloud::GeometryType::EDGE);
      //= cvo::downsample_lidar_points(false,
      //source_pc,
      //0.001);
    

    

    dataset.set_start_index(i+1);    
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZI>);
    if (dataset.read_next_lidar(target_pc) != 0) {
      std::cout<<"finish all files\n";
      break;
    }
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(target_pc, 5000, -1,
                                                                      cvo::CvoPointCloud::PointSelectionMethod::FULL));
    std::shared_ptr<cvo::CvoPointCloud> target_downsampled = cvo::voxel_downsample(target, 0.1, selected_pts, cvo::CvoPointCloud::GeometryType::EDGE);
    
    //= cvo::downsample_lidar_points(false,
    //                               target_pc,
    //                               0.001);
    


    std::cout<<"NUm of source pts is "<<source_downsampled->num_points()<<"\n";
    std::cout<<"NUm of target pts is "<<target_downsampled->num_points()<<"\n";

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    double this_time = 0;
    cvo_align.align(*source_downsampled, *target_downsampled, init_guess_inv, result, nullptr, &this_time);
    total_time += this_time;
    
    // get tf and inner product from cvo getter
    //double in_product = cvo_align.inner_product(source, target, result);

    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    //std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;
    
    
    // log accumulated pose

    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                 <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                 <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;
    
    std::cout<<"\n\n===========next frame=============\n\n";

        
    // transform the pc according to gt poses
    // std::cout << "first point local = " << pc->points[1].x << ", " << pc->points[1].y << ", " << pc->points[1].z << std::endl;
    //Eigen::Matrix4d T = gt_poses[i];
    //pcl::transformPointCloud(*pc, *pc, T);
    // std::cout << "T = " << std::endl;
    // std::cout << T << std::endl;
    // std::cout << "first point global= " << pc->points[1].x << ", " << pc->points[1].y << ", " << pc->points[1].z << std::endl;
    // viz

    //dataset.next_frame_index();
  }

  std::cout<<"time per frame is "<<total_time / double(i - start_frame + 1)<<std::endl;
  accum_output.close();

  return 0;
}
