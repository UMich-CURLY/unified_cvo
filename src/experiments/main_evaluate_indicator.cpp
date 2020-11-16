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
#include "graph_optimizer/Frame.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include <pcl/io/pcd_io.h>
//#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  std::cout<<"We are evaluating the indicator\n";
  string cvo_param_file(argv[1]);
  std::ofstream output(argv[2]);
  string sequence(argv[3]);

  string calib_file;
  // calib_file = "/home/cel/data/kitti/sequences/"+sequence+"/cvo_calib.txt"; 
  // cvo::Calibration calib(calib_file);
  calib_file = "/home/cel/data/tum/"+sequence+"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, 1);

  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  std::cout<<"ell_init = "<< ell_init<<std::endl;
  float ell_max = init_param.ell_max;
  init_param.ell_max = 2.5;//0.75;
  
  // read the source file
  int frame = 0;
  // string source_filename = "indicator_evaluation/rgbd_full/"+sequence+"_"+std::to_string(frame)+"_rgbd_full.pcd";
  string source_filename = "indicator_evaluation/rgbd_semi/"+sequence+"_"+std::to_string(frame)+"_rgbd_semi.pcd";
  // string source_filename = "indicator_evaluation/stereo_full/"+sequence+"_"+std::to_string(frame)+"_stereo_full.pcd";
  // string source_filename = "indicator_evaluation/stereo_semi_dense/"+sequence+"_"+std::to_string(frame)+"_stereo_semi_dense.pcd";
  std::cout<<"source_filename = "<<source_filename<<std::endl;   

  // // lidar
  // pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
  // if (pcl::io::loadPCDFile<pcl::PointXYZI> (source_filename, *source_pc) == -1) //* load the file
  // {
  //   PCL_ERROR ("Couldn't read file\n");
  //   return (-1);
  // }

  // stereo & RGB-D with color points
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (source_filename, *source_pc) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file\n");
    return (-1);
  }

  // stereo
  // std::shared_ptr<cvo::Frame> source(new cvo::Frame(0, source_pc, calib));
  // rgbd
  std::shared_ptr<cvo::Frame> source(new cvo::Frame(0, source_pc, true));

  auto source_fr = source->points();

  // calculate initial guess
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0.0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  
  // std::ofstream indicator_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_stereo_full_indicator_rotate_2.csv"); 
  // std::ofstream indicator_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_stereo_semi_indicator_rotate_2.csv"); 
  // std::ofstream indicator_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_rgbd_full_indicator_rotate_3.csv"); 
  std::ofstream indicator_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_rgbd_semi_indicator_rotate_y.csv"); 

  for (int angle = -180; angle <=-30; angle+=15){
    std::cout<<"\n\n===========next angle "<<angle<<"=============\n\n";
    // string target_filename = "indicator_evaluation/rgbd_full/"+sequence+"_"+std::to_string(frame)+"_rgbd_full.pcd";
    // string target_filename = "indicator_evaluation/stereo_full/"+sequence+"_"+std::to_string(frame)+"_stereo_full.pcd";
    // read the target file
    // std::cout<<"\n\n\n\n============================================="<<std::endl;
    // std::cout<<"target_filename = "<<target_filename<<std::endl;

    // // lidar
    // pcl::PointCloud<pcl::PointXYZI>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZI>);
    // if (pcl::io::loadPCDFile<pcl::PointXYZI> (target_filename, *target_pc) == -1) //* load the file
    // {
    //   PCL_ERROR ("Couldn't read file\n");
    //   return -1;
    // }
    //stereo
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    // if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (target_filename, *target_pc) == -1) //* load the file
    // {
    //   PCL_ERROR ("Couldn't read file\n");
    //   return -1;
    // }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // The same rotation matrix as before; theta radians around Z axis
    float theta = M_PI/180*angle; // The angle of rotation in radians
    transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

    // Executing the transformation
    pcl::transformPointCloud (*source_pc, *target_pc, transform);

    std::shared_ptr<cvo::Frame> target(new cvo::Frame(frame, target_pc, true));

    // set up frames
    auto target_fr = target->points();

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
 
    float indicator_value = cvo_align.indicator_value(source_fr, target_fr, init_guess_inv, result);
    
    std::cout<<"indicator value = "<<indicator_value<<std::endl;
    // std::cout<<"angle value = "<<function_angle_value<<std::endl;

    indicator_file << angle << " " << indicator_value<<"\n"<<std::flush;
    // function_angle_file << init_ell << " " << function_angle_value<<"\n"<<std::flush;
  }
  for (int angle = -28; angle <=28; angle+=2){
    std::cout<<"\n\n===========next angle "<<angle<<"=============\n\n";
    //stereo
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // The same rotation matrix as before; theta radians around Z axis
    float theta = M_PI/180*angle; // The angle of rotation in radians
    transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

    // Executing the transformation
    pcl::transformPointCloud (*source_pc, *target_pc, transform);

    std::shared_ptr<cvo::Frame> target(new cvo::Frame(frame, target_pc, true));

    // set up frames
    auto target_fr = target->points();

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
 
    float indicator_value = cvo_align.indicator_value(source_fr, target_fr, init_guess_inv, result);
    
    std::cout<<"indicator value = "<<indicator_value<<std::endl;

    indicator_file << angle << " " << indicator_value<<"\n"<<std::flush;
    
  }
  for (int angle = 30; angle <=180; angle+=15){
    
    std::cout<<"\n\n===========next angle "<<angle<<"=============\n\n";
    //stereo
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // The same rotation matrix as before; theta radians around Z axis
    float theta = M_PI/180*angle; // The angle of rotation in radians
    transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

    // Executing the transformation
    pcl::transformPointCloud (*source_pc, *target_pc, transform);

    std::shared_ptr<cvo::Frame> target(new cvo::Frame(frame, target_pc, true));

    // set up frames
    auto target_fr = target->points();

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
 
    float indicator_value = cvo_align.indicator_value(source_fr, target_fr, init_guess_inv, result);
    
    std::cout<<"indicator value = "<<indicator_value<<std::endl;

    indicator_file << angle << " " << indicator_value<<"\n"<<std::flush;
  }
  indicator_file.close();

  // std::ofstream indicator_translate_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_stereo_full_indicator_translate_3.csv"); 
  // std::ofstream indicator_translate_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_stereo_semi_indicator_translate_check.csv"); 
  // std::ofstream indicator_translate_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_rgbd_full_indicator_translate_3.csv"); 
  // std::ofstream indicator_translate_file("indicator_evaluation/value_history/"+sequence+"_"+std::to_string(frame)+"_rgbd_semi_indicator_translate_3.csv"); 

  // for (float distance = -2.5; distance <=2.5; distance+=0.05){
  //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    
  //   Eigen::Affine3f transform = Eigen::Affine3f::Identity();

  //   // Define a translation of ? meters on the x axis.
  //   transform.translation() << distance, 0.0, 0.0;

  //   // Executing the transformation
  //   pcl::transformPointCloud (*source_pc, *target_pc, transform);

  //   std::shared_ptr<cvo::Frame> target(new cvo::Frame(frame, target_pc, true));

  //   // set up frames
  //   auto target_fr = target->points();

  //   Eigen::Matrix4f result, init_guess_inv;
  //   init_guess_inv = init_guess.inverse();
    
  //   float indicator_value = cvo_align.indicator_value(source_fr, target_fr, init_guess_inv, result);
    
  //   std::cout<<"indicator value = "<<indicator_value<<std::endl;
    
  //   indicator_translate_file << distance << " " << indicator_value<<"\n"<<std::flush;
      
  //   std::cout<<"\n\n===========next distance "<<distance<<"=============\n\n";

  // }
  // indicator_translate_file.close();

  return 0;
}
