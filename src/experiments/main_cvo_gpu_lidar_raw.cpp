#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
#include "dataset_handler/KittiHandler.hpp"
//#include "graph_optimizer/Frame.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
//#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  std::cout<<"start\n";
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::KittiHandler kitti(argv[1], 1);
  int total_iters = kitti.get_total_number();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  kitti.set_start_index(start_frame);
  int max_num = std::stoi(argv[5]);

  accum_output <<"1 0 0 0 0 1 0 0 0 0 1 0\n";

  std::cout<<"new cvo_align\n";
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_max = init_param.ell_max;
  init_param.ell_init = 0.51;//0.51;
  init_param.ell_max = 1.5;//0.75;
  cvo_align.write_params(&init_param);
  
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0.0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  // start the iteration

  //std::shared_ptr<cvo::Frame> source(new cvo::Frame(start_frame, source_pc,
  //                                                  calib));
  //0.2));
  double total_time = 0;
  int i = start_frame;
  for (; i<min(total_iters, start_frame+max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
    //std::vector<int> semantics_source;
    kitti.read_next_lidar(source_pc);
    
    std::cout<<"read next lidar\n"; 
    cvo::CvoPointCloud source(source_pc, 5000, 64);

    kitti.next_frame_index();
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZI>);
    if (kitti.read_next_lidar(target_pc) != 0) {
      std::cout<<"finish all files\n";
      break;
    }
    cvo::CvoPointCloud target(target_pc, 5000, 64);
    //std::shared_ptr<cvo::Frame> target(new cvo::Frame(i+1, target_pc, calib));

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;
    std::cout<<"NUm of source pts is "<<source.num_points()<<"\n";
    std::cout<<"NUm of target pts is "<<target.num_points()<<"\n";

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    double this_time = 0;
    cvo_align.align(source, target, init_guess_inv, result, &this_time);
    total_time += this_time;
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product(source, target, result);

    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
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
   
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_max = ell_max;
      cvo_align.write_params(&init_param);
      
    } //else if (i < start_frame + 20)  {
      //init_param.ell_init =  1.0;
      //init_param.ell_max = 1.0;
      //cvo_align.write_params(&init_param);

      
    //}

  }

  std::cout<<"time per frame is "<<total_time / double(i - start_frame + 1)<<std::endl;
  accum_output.close();

  return 0;
}
