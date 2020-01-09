
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include "utils/CvoPointCloud.hpp"
#include "cvo/Cvo.cuh"

using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  string data_folder (argv[1]);
  std::ofstream relative_output(argv[2]);
  std::ofstream accum_output(argv[3]);
  int max_num = std::stoi(argv[4]);
  
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
      if (total_num == max_num )
        break;
      // cout <<"reading "<< current_file << endl; 
    }
  }
  
  cvo::CvoGPU cvo_align("cvo_params.txt");
  Eigen::Affine3f init_guess;  // from source frame to the target frame
  init_guess.matrix().setIdentity();
  init_guess.matrix()(2,3)=0.75;

  // start the iteration
  for (int i = 0; i<total_num ; i++) {
    
    // calculate initial guess
    std::cout<<"\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<std::endl;

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;
    cvo::CvoPointCloud source_fr(files[i]);
    cvo::CvoPointCloud target_fr(files[i+1]);

    Eigen::Affine3f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    cvo_align.align(source_fr, target_fr, init_guess_inv, result);
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product(source_fr, target_fr, result);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result.matrix() <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = init_guess*result;
    std::cout<<"accum tf: \n"<<init_guess<<std::endl;
    

    // log relative pose
    Eigen::Matrix4f relative_mat = result.matrix();
    relative_output << relative_mat(0,0)<<" "<<relative_mat(0,1)<<" "<<relative_mat(0,2)<<" "<<relative_mat(0,3)<<" "
                <<relative_mat(1,0)<<" " <<relative_mat(1,1)<<" "<<relative_mat(1,2)<<" "<<relative_mat(1,3)<<" "
                <<relative_mat(2,0)<<" " <<relative_mat(2,1)<<" "<<relative_mat(2,2)<<" "<<relative_mat(2,3);
    relative_output<<"\n";
    relative_output<<std::flush;

    // log accumulated pose
    Eigen::Matrix4f accum_mat = init_guess.matrix();
    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;


  }

  relative_output.close();
  accum_output.close();

  return 0;
}
