
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
#include "graph_optimizer/Frame.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "utils/Calibration.hpp"

using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  int num_class= 0;
  string n_class_str;
  if (argc > 8) {
    n_class_str = argv[8];
    num_class = stoi(n_class_str);
  }
  
  int mode = stoi(argv[1]); // 0 for online generate 1 for read txt
  string pth (argv[2]);
  string txt_pth = pth + "/" + (argv[3]);
  string calib_name = pth + "/" + (argv[4]);
  std::ofstream output_file(argv[5]);
  int start_frame = stoi(argv[6]);
  double in_product_th = stof(argv[7]);
  int kf_step = 5;
  int total_num = 0;
  
  std::ofstream in_product_output_file("inner_product_kf_all.txt");

  vector<string> files;
  std::cout<<"pth: "<<pth<<std::endl;
  cvo::KittiHandler kitti(pth, 0);
  cvo::Calibration calib(calib_name);

  kitti.set_start_index(start_frame);

  if(mode==0){
    total_num = kitti.get_total_number();
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
  Eigen::Matrix4f tf_kf_im1 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_im1_im2 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f accum_tf = Eigen::Matrix4f::Identity();
  std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> accum_tf_list;
  accum_tf_list.push_back(Eigen::Matrix4f::Identity());
  std::vector<int> kf_id_list;
  kf_id_list.push_back(start_frame);
  std::vector<std::shared_ptr<cvo::Frame>> all_frames_since_last_keyframe;
  std::vector<float> in_product_list;
  in_product_list.push_back(1);
  cv::Mat left, right;
  double in_product_base = 1.0;
  double in_product_ratio = 0.0;
  std::shared_ptr<cvo::Frame> source_frame;
  std::shared_ptr<cvo::Frame> target_frame;

  // create first frame for kf
  if(kitti.read_next_stereo(left, right) == 0){
    source_frame = make_shared<cvo::Frame>(start_frame, left, right, calib);
    all_frames_since_last_keyframe.push_back(source_frame);
  }
  bool in_kf_init_process = false;
  for (int i = start_frame+1; i<total_num ; i++) {

    int cur_kf = kf_id_list.back();
    int prev_kf = kf_id_list.rbegin()[1];

    // if we are in the kf_init_process, calculate kf-1 to kf to find initialization of tf
    if(in_kf_init_process){
      // redo kf-1 and kf
      i = cur_kf;
      cur_kf -= 1;
      // init_guess.setIdentity();
      // init_guess.matrix()(2,3)=-0.75;

      // if(cur_kf-prev_kf>1)
      //   init_guess = (accum_tf_list[prev_kf-start_frame].inverse()*accum_tf_list[prev_kf+1-start_frame]).inverse();
      // else
      //   init_guess.setIdentity();
    }
    // else{
    // calculate initial guess from previous frames
    tf_kf_im1 = accum_tf_list[cur_kf-start_frame].inverse()*accum_tf_list[i-1-start_frame];
    if(i-start_frame>1)
      tf_im1_im2 = accum_tf_list[i-2-start_frame].inverse()*accum_tf_list[i-1-start_frame];
    init_guess = (tf_kf_im1*tf_im1_im2).inverse();
    // }
    
    std::cout<<"\n============================================="<<std::endl;
    std::cout<<"Aligning "<<cur_kf<<" and "<<i<<std::endl;
    if(mode==0){
      if(!in_kf_init_process){  // if we are not redoing kf and kf-1, add new frame to the list
        if(kitti.read_next_stereo(left, right) == 0){
          target_frame = make_shared<cvo::Frame>(start_frame, left, right, calib);
          all_frames_since_last_keyframe.push_back(target_frame);
          auto& source_fr = all_frames_since_last_keyframe.front()->points(); // keyframe
          auto& target_fr = target_frame->points();
          cvo_align.set_pcd(source_fr, target_fr, init_guess, true);
          cvo_align.align();
        }
      }
      else{ // if we are redoing kf and kf-1
          auto& source_fr = all_frames_since_last_keyframe.rbegin()[1]->points();
          auto& target_fr = all_frames_since_last_keyframe.back()->points();
          cvo_align.set_pcd(source_fr, target_fr, init_guess, true);
          cvo_align.align();
      }
    }
    else if(mode==1){
      // std::cout<<"reading "<<files[cur_kf]<<std::endl;
      cvo::CvoPointCloud source_fr;
      cvo::CvoPointCloud target_fr;
      source_fr.read_cvo_pointcloud_from_file(files[cur_kf]);
      target_fr.read_cvo_pointcloud_from_file(files[i]);
      const cvo::CvoPointCloud &const_source_fr = source_fr;
      const cvo::CvoPointCloud &const_target_fr = target_fr;
      
      cvo_align.set_pcd(const_source_fr, const_target_fr, init_guess, true);
      cvo_align.align();
    }
    
    
    

    init_guess= cvo_align.get_transform();
    double in_product = cvo_align.inner_product();
    double in_product_normalized = cvo_align.inner_product_normalized();
    int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    Eigen::Matrix4f result = init_guess.matrix();
    std::cout<<"The inner product between "<<cur_kf <<" and "<< i <<" is "<<in_product<<"\n";
    std::cout<<"Transform is \n";
    std::cout<<cvo_align.get_transform().matrix() <<"\n\n";
    

    

    // end the kf_init_process
    if(in_kf_init_process){
      in_kf_init_process=false;
      // clear all_frames_since_last_keyframe
      std::shared_ptr<cvo::Frame> temp_frame = all_frames_since_last_keyframe.back();
      all_frames_since_last_keyframe.clear();
      all_frames_since_last_keyframe.push_back(temp_frame);
    }
    // in_product_ratio = in_product/in_product_base;
    // std::cout<<"The inner product ratio for "<<i<<" is "<<in_product_ratio<<std::endl;

    // if it's kf and kf+1, record inner product for calculating ratios
    // if(all_frames_since_last_keyframe.size()==2){
    //   in_product_base = cvo_align.inner_product();
    // }
    // start the kf init process :)
    else if(in_product < in_product_th){
    // else if(i-kf_id_list.back()==kf_step){
      in_kf_init_process = true;
      // add new kf to the list
      kf_id_list.push_back(i);
      std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
      std::cout<<"start the kf init process"<<std::endl;
      std::cout<<"will redo "<<i-1<<" and "<< i << std::endl;
      std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;

      continue;
    }

    in_product_output_file<<non_zeros_in_A<<" "<<in_product<<" "<<in_product_normalized<<"\n";
    in_product_output_file<<std::flush;


    // record accum_tf for future initialization
    std::cout<<"multiplying tf with "<<cur_kf<<std::endl;
    accum_tf = accum_tf_list[cur_kf-start_frame]*cvo_align.get_transform().matrix();
    accum_tf_list.push_back(accum_tf);
    std::cout<<"adding "<<accum_tf_list.size()-1<<" to accum_tf"<<std::endl;
    result = accum_tf;
    output_file << result(0,0)<<" "<<result(0,1)<<" "<<result(0,2)<<" "<<result(0,3)<<" "
                <<result(1,0)<<" " <<result(1,1)<<" "<<result(1,2)<<" "<<result(1,3)<<" "
                <<result(2,0)<<" " <<result(2,1)<<" "<<result(2,2)<<" "<<result(2,3);
     //output_file << result.block<3,4>(0,0);
    output_file<<"\n";
    output_file<<std::flush;

    
  }

  output_file.close();
  in_product_output_file.close();
  return 0;
}
