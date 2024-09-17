#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
//#include "dataset_handler/TartanAirHandler.hpp"
//#include "graph_optimizer/Frame.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/CvoParams.hpp"
#include "utils/ImageRGBD.hpp"
#include "viewer/viewer.h"

using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::string in_pcd_folder(argv[1]);
  
  string cvo_param_file(argv[2]);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  int last_frame = std::stoi(argv[5]);
  int is_stacking_results = std::stoi(argv[6]);
  int is_visualize = std::stoi(argv[7]);

  int num_frames = last_frame - start_frame + 1;
  
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_decay_rate = init_param.ell_decay_rate;
  int ell_decay_start = init_param.ell_decay_start;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  cvo_align.write_params(&init_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  //cvo::cvo cvo_align_cpu("/home/rayzhang/outdoor_cvo/cvo_params/cvo_params.txt");
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  Eigen::Quaternionf q(accum_mat.block<3,3>(0,0));
  accum_output<<accum_mat(0,3)<<" "<<accum_mat(1,3)<<" "<<accum_mat(2,3)<<" "; 
  accum_output<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
  accum_output.flush();
  // start the iteration


  std::string fname = in_pcd_folder + "/" + std::to_string(start_frame) + ".pcd";
  pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd(new pcl::PointCloud<cvo::CvoPoint>);
  pcl::io::loadPCDFile<cvo::CvoPoint> (fname, *raw_pcd);
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*raw_pcd));
  for (int j = 0; j < source->size(); j++) {
    auto & p = source->point_at(j);
    p.features[0] = static_cast<float>(p.b) / 255.0;
    p.features[1] = static_cast<float>(p.g) / 255.0;
    p.features[2] = static_cast<float>(p.r) / 255.0;
  }
  std::cout<<"read source cvo point cloud\n";  
  std::cout<<"source point cloud size is "<<source->size()<<std::endl;

  cvo::CvoPointCloud pc_all(FEATURE_DIMENSIONS, NUM_CLASSES);
  pc_all += *source;


  std::unique_ptr<perl_registration::Viewer> viewer;
  if (is_visualize)  {
    viewer = std::make_unique<perl_registration::Viewer>();
    std::string s("title");
    viewer->addOrUpdateText (s,
                             0,
                             0,
                             "title");
  }
  
  for (int i = start_frame+1; i< num_frames ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i-1<<" and "<<i<<" with GPU "<<std::endl;

    fname = in_pcd_folder + "/" + std::to_string(i)+ ".pcd"; 
    pcl::PointCloud<cvo::CvoPoint>::Ptr raw_pcd_target(new pcl::PointCloud<cvo::CvoPoint>);
    pcl::io::loadPCDFile<cvo::CvoPoint> (fname, *raw_pcd_target);
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*raw_pcd_target));
  for (int j = 0; j < target->size(); j++) {
    auto & p = target->point_at(j);
    p.features[0] = static_cast<float>(p.b) / 255.0;
    p.features[1] = static_cast<float>(p.g) / 255.0;
    p.features[2] = static_cast<float>(p.r) / 255.0;
  }
    


    if (i == start_frame+1){
        std::cout<<"Write first pcd\n";
        //target->write_to_color_pcd(std::to_string(i+1)+".pcd");
    }
    //std::cout<<"First point is "<<target->at(0).transpose()<<std::endl;

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
    std::cout<<std::flush;
    cvo_align.align(*source, *target, init_guess_inv, result);
    
    // get tf and inner product from cvo getter
    //double in_product = cvo_align.inner_product_cpu(*source, *target, result, ell_init);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    //std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    //init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;

    if (is_visualize) {
      Eigen::Matrix<double, 3, 4, Eigen::RowMajor> m = accum_mat.block<3,4>(0,0).cast<double>();
    //std::cout<<__func__<<": send \n"<<m.col(3).transpose()<<" for index "<<j<<" to viewer\n";
      viewer->drawTrajectory(m);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointSeg_to_PointXYZRGB<cvo::CvoPoint::FEATURE_DIMENSION,
                                   cvo::CvoPoint::LABEL_DIMENSION,
                                   pcl::PointXYZRGB>(*raw_pcd_target, *cloud);
      pcl::transformPointCloud (*cloud, *cloud, accum_mat);
      std::string viewer_id = std::to_string(i);
      viewer->updateColorPointCloud(*cloud, viewer_id);
    }
    
    
    
    // log accumulated pose

    // accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
    //             <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
    //             <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    // accum_output<<"\n";
    // accum_output<<std::flush;
    
    Eigen::Quaternionf q(accum_mat.block<3,3>(0,0));
    //accum_output<<vstrRGBName[i]<<" ";
    accum_output<<accum_mat(0,3)<<" "<<accum_mat(1,3)<<" "<<accum_mat(2,3)<<" "; 
    accum_output<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
    accum_output.flush();

    if (is_stacking_results) {
      cvo::CvoPointCloud target_transformed(FEATURE_DIMENSIONS, NUM_CLASSES);
      cvo::CvoPointCloud::transform(result, *target, target_transformed);
      pc_all += target_transformed;
    }

    std::cout<<"\n\n===========next frame=============\n\n";
   
    source = target;
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_decay_rate = ell_decay_rate;
      init_param.ell_decay_start = ell_decay_start;
      
      cvo_align.write_params(&init_param);
      
    }


  }

  if (is_stacking_results) {
    //pc_all.write_to_color_pcd("stacked_tracking.pcd");
  }

  accum_output.close();


  if (is_visualize) {
    while (!viewer->wasStopped()) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000000));
    }
    
  }

  

  return 0;
}
