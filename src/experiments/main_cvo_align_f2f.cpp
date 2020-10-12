
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include "utils/CvoPointCloud.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/ACvo.hpp"
#include "graph_optimizer/Frame.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "utils/Calibration.hpp"

using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  int num_class= 0;
  string use_semantic_str = "";
  string use_lidar_str = "";
  bool use_semantic = false;
  if (argc > 9) {
    num_class = stoi(argv[9]);
    std::cout<<"num_classes: "<<num_class<<std::endl;
    use_semantic_str = "_semantic";
    use_semantic = true;
  }
  
  int mode = stoi(argv[1]); // 0 for online generate 1 for read txt
  string pth (argv[2]);
  string txt_pth = pth + "/" + (argv[3]);
  string calib_name = pth + "/" + (argv[4]);
  std::ofstream output_file(argv[5]);
  int start_frame = stoi(argv[6]);
  string dataset_num = argv[7];
  int data_type = stoi(argv[8]); // 0 for stereo, 1 for lidar
  
  if(data_type==1){
    use_lidar_str = "_lidar";
  }
  

  int total_num = 0;
  
  std::ofstream accum_tf_output_file("results/cvo_f2f_tracking_"+dataset_num+use_lidar_str+use_semantic_str+".txt");
  std::ofstream in_product_output_file("results/inner_product_all_"+dataset_num+use_lidar_str+use_semantic_str+".txt");
  std::ofstream num_points_output_file("results/num_points_"+dataset_num+use_lidar_str+use_semantic_str+".txt");

  vector<string> files;
  std::cout<<"pth: "<<pth<<std::endl;
  cvo::KittiHandler kitti_stereo(pth, data_type), kitti_lidar(pth, data_type);
  cvo::Calibration calib;

  if (data_type==0){
    calib.read_file(calib_name);
    kitti_stereo.set_start_index(start_frame);
  }
  else if(data_type==1){
    kitti_lidar.set_start_index(start_frame);

    #undef FEATURE_DIMENSIONS
    #define FEATURE_DIMENSIONS 1
  }    
  
  if(mode==0){
    if (data_type==0){
      total_num = kitti_stereo.get_total_number();
    }
    else if(data_type==1){
      total_num = kitti_lidar.get_total_number();
    }      
  }
  else if(mode==1){
    // cycle through the directory
    std::cout<<"reading cvo points from txt files..."<<std::endl;
    for(auto & p : boost::filesystem::directory_iterator( txt_pth ) )
    {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
      if (is_regular_file(p.path())) {
      // assign current file name to current_file and echo it out to the console.
      string current_file = p.path().string();
      files.push_back(txt_pth + to_string(total_num) + ".txt" );
      total_num += 1;
      // cout <<"reading "<< current_file << endl; 
    }
  }
  }
 
  
  cvo::cvo cvo_align;
  Eigen::Affine3f init_guess;
  init_guess.matrix().setIdentity();
  // init_guess.matrix()(2,3)=0.75;
  Eigen::Matrix4f result = init_guess.matrix();
  result = init_guess.matrix();
  for(int i=0; i<start_frame+1; ++i){
    accum_tf_output_file << result(0,0)<<" "<<result(0,1)<<" "<<result(0,2)<<" "<<result(0,3)<<" "
                  <<result(1,0)<<" " <<result(1,1)<<" "<<result(1,2)<<" "<<result(1,3)<<" "
                  <<result(2,0)<<" " <<result(2,1)<<" "<<result(2,2)<<" "<<result(2,3);
    accum_tf_output_file<<"\n";
    accum_tf_output_file<<std::flush;
  }
  Eigen::Matrix4f tf_kf_im1 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_im1_im2 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f accum_tf = Eigen::Matrix4f::Identity();
  std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> accum_tf_list;
  accum_tf_list.push_back(init_guess.matrix());
  std::vector<float> in_product_list;
  in_product_list.push_back(1);
  cv::Mat left, right;
  std::vector<float> semantics;
  std::shared_ptr<cvo::Frame> source_frame;
  std::shared_ptr<cvo::Frame> target_frame;
  std::shared_ptr<cvo::Frame> pcl_source_frame;
  std::shared_ptr<cvo::Frame> pcl_target_frame;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc (new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<int> semantics_lidar;

  // load the first image
  if(mode==0){
    if(data_type==0){
      if(use_semantic){
        // create first frame for kf
        if (kitti_stereo.read_next_stereo(left, right, num_class, semantics ) == 0) {
          kitti_stereo.next_frame_index();
          std::shared_ptr<cvo::Frame> temp_source(new cvo::Frame(start_frame, left, right, num_class, semantics, calib ));
          source_frame = temp_source; 
        }
      }
      else{
        // create first frame for kf
        if(kitti_stereo.read_next_stereo(left, right) == 0){
          kitti_stereo.next_frame_index();
          std::shared_ptr<cvo::Frame> temp_source(new cvo::Frame(start_frame, left, right, calib ));
          source_frame = temp_source;
        }
      }
    }
    else if(data_type==1){
      if(use_semantic){
        // create first frame for kf
        if (kitti_lidar.read_next_lidar(pc, semantics_lidar ) == 0) {
          kitti_lidar.next_frame_index();
          std::shared_ptr<cvo::Frame> temp_pcl_source(new cvo::Frame(start_frame, pc, semantics_lidar, calib ));
          pcl_source_frame = temp_pcl_source;
        }
      }
      else{
        // create first frame for kf
        if(kitti_lidar.read_next_lidar(pc) == 0){
          kitti_lidar.next_frame_index();
          std::shared_ptr<cvo::Frame> temp_pcl_source(new cvo::Frame(start_frame, pc, calib ));
          pcl_source_frame = temp_pcl_source;
        }
      }
    }
  }

  // num_points
  num_points_output_file << "0 "<< pcl_source_frame->points().num_points();
  num_points_output_file << "\n";
  num_points_output_file<<std::flush;
  

  // start the iteration
  for (int i = start_frame+1; i<total_num ; i++) {
  // for (int i = start_frame+1; i<2 ; i++) {
    
    // calculate initial guess
    if(accum_tf_list.size()>=2){
      init_guess = (accum_tf_list[i-2-start_frame].inverse()*accum_tf_list[i-1-start_frame]).inverse();
    }
    else{
      init_guess.setIdentity();
      init_guess.matrix()(2,3)=0;
    }

    std::cout<<"\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i-1<<" and "<<i<<std::endl;

    // load next image
    // mode 0: online point cloud generation, mode 1: load point cloud from txt
    if(mode==0){
      if(data_type==0){
        if(use_semantic){
          // load image and semantic and create point cloud into cvo::Frame
          if (kitti_stereo.read_next_stereo(left, right, num_class, semantics ) == 0) {
            kitti_stereo.next_frame_index();
            std::shared_ptr<cvo::Frame> temp_target(new cvo::Frame(i, left, right, num_class, semantics, calib ));
            target_frame = temp_target; 
          }
        }
        else{
          // load image and create point cloud into cvo::Frame
          if(kitti_stereo.read_next_stereo(left, right) == 0){
            kitti_stereo.next_frame_index();
            std::shared_ptr<cvo::Frame> temp_target(new cvo::Frame(i, left, right, calib ));
            target_frame = temp_target; 
          }
        }

        auto& source_fr = source_frame->points(); // keyframe
        auto& target_fr = target_frame->points();
        cvo_align.set_pcd(source_fr, target_fr, init_guess, true);
        if(i==start_frame+1){
          cvo_align.ell = 1;
          cvo_align.ell_max = 1.2;
          std::cout<<"initialize ell to: "<< cvo_align.ell<<std::endl;
        }
        cvo_align.align();
      }
      else if(data_type==1){
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc (new pcl::PointCloud<pcl::PointXYZI>);
        std::vector<int> semantics_lidar;
        if(use_semantic){
          // create first frame for kf
          if (kitti_lidar.read_next_lidar(pc, semantics_lidar ) == 0) {
            kitti_lidar.next_frame_index();
            std::shared_ptr<cvo::Frame> temp_pcl_target(new cvo::Frame(i, pc, semantics_lidar, calib ));
            pcl_target_frame = temp_pcl_target; 
          }
        }
        else{
          // create first frame for kf
          if(kitti_lidar.read_next_lidar(pc) == 0){
            kitti_lidar.next_frame_index();
            std::shared_ptr<cvo::Frame> temp_pcl_target(new cvo::Frame(i, pc, calib ));
            pcl_target_frame = temp_pcl_target; 
          }
        }

        // num_points
        num_points_output_file << std::to_string(i)<< " "<< pcl_source_frame->points().num_points();
        num_points_output_file << "\n";
        num_points_output_file<<std::flush;

        auto& source_fr = pcl_source_frame->points(); // keyframe
        auto& target_fr = pcl_target_frame->points();
        cvo_align.set_pcd(source_fr, target_fr, init_guess, true);
        if(i==start_frame+1){
          cvo_align.ell = 1;
          cvo_align.ell_max = 1.2;
          std::cout<<"initialize ell to: "<< cvo_align.ell<<std::endl;
        }
        cvo_align.align();
      }
        

        
    }
    else if(mode==1){
      // std::cout<<"reading "<<files[cur_kf]<<std::endl;
      cvo::CvoPointCloud source_fr;
      cvo::CvoPointCloud target_fr;
      source_fr.read_cvo_pointcloud_from_file(files[i-1+start_frame]);
      target_fr.read_cvo_pointcloud_from_file(files[i+start_frame]);
      const cvo::CvoPointCloud &const_source_fr = source_fr;
      const cvo::CvoPointCloud &const_target_fr = target_fr;
      
      cvo_align.set_pcd(const_source_fr, const_target_fr, init_guess, true);
      if(i==start_frame+1){
          cvo_align.ell = 1;
          cvo_align.ell_max = 1.2;
          std::cout<<"initialize ell to: "<< cvo_align.ell<<std::endl;
        }
      cvo_align.align();
    }

    
    
    // get tf and inner product from cvo getter
    init_guess= cvo_align.get_transform();
    double in_product = cvo_align.inner_product();
    double in_product_normalized = cvo_align.inner_product_normalized();
    int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is \n";
    std::cout<<cvo_align.get_transform().matrix() <<"\n\n";

    // write inner product into txt
    in_product_output_file<<non_zeros_in_A<<" "<<in_product<<" "<<in_product_normalized<<"\n";
    in_product_output_file<<std::flush;

    // append accum_tf_list for future initialization
    accum_tf = accum_tf_list[i-1-start_frame]*cvo_align.get_transform().matrix();
    accum_tf_list.push_back(accum_tf);
    std::cout<<"adding "<<accum_tf_list.size()-1<<" to accum_tf"<<std::endl;
    std::cout<<"accum tf: \n"<<accum_tf<<std::endl;
    

    // log relative pose
    result = cvo_align.get_transform().matrix();
    output_file << result(0,0)<<" "<<result(0,1)<<" "<<result(0,2)<<" "<<result(0,3)<<" "
                <<result(1,0)<<" " <<result(1,1)<<" "<<result(1,2)<<" "<<result(1,3)<<" "
                <<result(2,0)<<" " <<result(2,1)<<" "<<result(2,2)<<" "<<result(2,3);
    output_file<<"\n";
    output_file<<std::flush;

    // log accumulated pose
    result = accum_tf;
    accum_tf_output_file << result(0,0)<<" "<<result(0,1)<<" "<<result(0,2)<<" "<<result(0,3)<<" "
                <<result(1,0)<<" " <<result(1,1)<<" "<<result(1,2)<<" "<<result(1,3)<<" "
                <<result(2,0)<<" " <<result(2,1)<<" "<<result(2,2)<<" "<<result(2,3);
    accum_tf_output_file<<"\n";
    accum_tf_output_file<<std::flush;

    // move source frame to target frame
    // source_frame = target_frame;
    pcl_source_frame = pcl_target_frame;
  }

  output_file.close();
  accum_tf_output_file.close();
  in_product_output_file.close();
  num_points_output_file.close();
  return 0;
}
