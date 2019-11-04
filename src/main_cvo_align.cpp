
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include "utils/CvoPointCloud.hpp"
#include "cvo/Cvo.hpp"

using namespace std;
using namespace boost::filesystem;



int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  int num_class= 0;
  string n_class_str;
  if (argc > 5) {
    n_class_str = argv[5];
    num_class = stoi(n_class_str);
  }
  
  
  path p (argv[1] );
  std::ofstream output_file(argv[2]);
  int start_frame = stoi(argv[3]);
  int kf_size = stoi(argv[4]);
  int num_frames;
  
  
  vector<string> files;
  // cycle through the directory
  int total_num = 0;
  for(auto & p : boost::filesystem::directory_iterator( p ) )
  {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (is_regular_file(p.path())) {
      // assign current file name to current_file and echo it out to the console.
      string current_file = p.path().string();
      files.push_back(string(argv[1]) + "/" + to_string(total_num) + ".txt" );
      total_num += 1;
      //cout <<"reading "<< current_file << endl; 

    }
  }

  // std::cout<<"Just read file names\n";
  // std::vector<cvo::CvoPointCloud> pc_vec(files.size());
  // int i = 0;
  // for (auto &f:  files) {
  //   std::cout<<"Reading "<<f<<std::endl;
  //   //dso::read_cvo_pointcloud_from_file<dso::CvoTrackingPoints>(f, all_pts[i]);
  //   pc_vec[i].read_cvo_pointcloud_from_file(f);
  //   i ++;
  //   if (i == num_frames) break;
  // }
  // std::cout<<"Just reading  names\n";

  
  cvo::cvo cvo_align;
  Eigen::Affine3f init_guess;
  init_guess.matrix().setIdentity();
  // init_guess.matrix()(2, 3) = 0.75;
  Eigen::Matrix4f tf_kf_minus_2= Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_kf_minus_1= Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_kf_init_guess = Eigen::Matrix4f::Identity();
  int kf_id = 0;
  for (int i = start_frame; i<total_num ; i++) {

    if(i-kf_id==kf_size){
            kf_id = i;
            tf_kf_minus_1 = cvo_align.get_transform().matrix();
            tf_kf_init_guess = tf_kf_minus_1*tf_kf_minus_2.inverse();
            tf_kf_init_guess = tf_kf_init_guess.inverse().eval();

            std::cout<<"tf_3to0: \n"<<tf_kf_minus_2<<std::endl;
            std::cout<<"tf_4to0: \n"<<tf_kf_minus_1<<std::endl;
            std::cout<<"tf_4to3: \n"<<tf_kf_minus_1*tf_kf_minus_2.inverse()<<std::endl;
            std::cout<<"init guess for new kf is: \n"<<tf_kf_init_guess<<std::endl;

            init_guess =  tf_kf_init_guess;
            continue;
        }
        else if(i-kf_id==kf_size-1){
            tf_kf_minus_2 = cvo_align.get_transform().matrix();
        }

    cvo::CvoPointCloud kf;
    cvo::CvoPointCloud fr;
    
    kf.read_cvo_pointcloud_from_file(files[kf_id]);
    fr.read_cvo_pointcloud_from_file(files[i]);

    std::cout<<"\n=============================================\nat"<<i<<"\n iter";

    cvo_align.set_pcd(kf, fr, init_guess, true);
    cvo_align.align();

    init_guess= cvo_align.get_accum_transform();
    Eigen::Matrix4f result = init_guess.matrix();
    std::cout<<"\n The inner product between "<<i <<" and "<< i+1 <<" is "<<cvo_align.inner_product()<<"\n";
    std::cout<<"Transform is \n";
    std::cout<<cvo_align.get_transform().matrix() <<"\n\n";
     output_file << result(0,0)<<" "<<result(0,1)<<" "<<result(0,2)<<" "<<result(0,3)<<" "
                <<result(1,0)<<" " <<result(1,1)<<" "<<result(1,2)<<" "<<result(1,3)<<" "
                <<result(2,0)<<" " <<result(2,1)<<" "<<result(2,2)<<" "<<result(2,3);
     //output_file << result.block<3,4>(0,0);
    output_file<<"\n";
    output_file<<std::flush;
  }
  

  output_file.close();
  
  return 0;
}
