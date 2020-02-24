#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>

#include "utils/CvoPointCloud.hpp"
#include "cvo/AdaptiveCvoGPU.hpp"
#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  string data_folder (argv[1]);
  string cvo_param_file(argv[2]);
  std::ofstream relative_output(argv[3]);
  std::ofstream accum_output(argv[4]);
  int start_frame = std::stoi(argv[5]);
  int max_num = std::stoi(argv[6]);
  
  vector<string> files;
  // cycle through the directory
  std::cout<<"reading cvo points from txt files..."<<std::endl;
  int total_num = 0;
  for(auto & p : boost::filesystem::directory_iterator( data_folder ) ) {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (is_regular_file(p.path())) {
      // assign current file name to current_file and echo it out to the console.
      string current_file = p.path().string();
      files.push_back(data_folder + to_string(total_num) + ".txt" );
      total_num += 1;
      //if (total_num == max_num )
      //  break;
      // cout <<"reading "<< current_file << endl; 
    }
  }
  
  cvo::AdaptiveCvoGPU cvo_align(cvo_param_file );
  cvo::cvo cvo_align_cpu("/home/rayzhang/outdoor_cvo/cvo_params/cvo_params.txt");
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;

  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  // start the iteration
  for (int i = start_frame; i<min(total_num, start_frame+max_num) ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::string f1 = data_folder + "/" + std::to_string(i)+".txt";
    std::string f2 = data_folder + "/" + std::to_string(i+1)+".txt";
    std::cout<<"Aligning "<<f1<<" and "<<f2<<" with GPU "<<std::endl;

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;
    cvo::CvoPointCloud source_fr;
    source_fr.read_cvo_pointcloud_from_file(f1);
    cvo::CvoPointCloud target_fr;//(files[i+1]);
    target_fr.read_cvo_pointcloud_from_file(f2);

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source_fr.num_points(), target_fr.num_points());
    std::cout<<std::flush;
    cvo_align.align(source_fr, target_fr, init_guess_inv, result);
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product(source_fr, target_fr, result);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;
    
    
    // log relative pose
    Eigen::Matrix4f relative_mat = result;
    relative_output << relative_mat(0,0)<<" "<<relative_mat(0,1)<<" "<<relative_mat(0,2)<<" "<<relative_mat(0,3)<<" "
                <<relative_mat(1,0)<<" " <<relative_mat(1,1)<<" "<<relative_mat(1,2)<<" "<<relative_mat(1,3)<<" "
                <<relative_mat(2,0)<<" " <<relative_mat(2,1)<<" "<<relative_mat(2,2)<<" "<<relative_mat(2,3);
    relative_output<<"\n";
    relative_output<<std::flush;
   
    // log accumulated pose

    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;
    
    /*
    std::cout<<"\n---------------------------------------------------"<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with CPU "<<std::endl;
    Eigen::Affine3f result_cpu,init_guess_inv_cpu;
    init_guess_inv_cpu = init_guess_cpu.inverse();
    cvo_align_cpu.set_pcd(source_fr, target_fr, init_guess_inv_cpu, true);
    cvo_align_cpu.align();
    result_cpu = cvo_align_cpu.get_transform();
    // get tf and inner product from cvo getter
    double in_product_cpu = cvo_align_cpu.inner_product(source_fr, target_fr, result_cpu);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The cpu inner product between "<<i <<" and "<< i+1 <<" is "<<in_product_cpu<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform cpu is "<<result_cpu.matrix() <<"\n\n";
    */
    // append accum_tf_list for future initialization
    //init_guess_cpu = init_guess_cpu*result_cpu;
    //std::cout<<"accum tf: \n"<<init_guess_cpu.matrix()<<std::endl;
    
    std::cout<<"\n\n===========next frame=============\n\n";
   


  }

  relative_output.close();
  accum_output.close();

  return 0;
}
