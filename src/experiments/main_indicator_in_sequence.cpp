#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  cvo::KittiHandler kitti(argv[1], 0);
  int total_iters = kitti.get_total_number();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1]) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  string sequence(argv[3]);
  string method(argv[4]);
  std::cout<<"\n\nstart indicator evaluation for "<<sequence<<"\n\n";

  // output file
  std::ofstream indicator_file("indicator_evaluation/value_history_sequence/"+sequence+"/"+method+".csv"); 
  std::cout<<"the indicator_file has been created\n";
  
  // get transformation file
  string transformation_file(argv[5]);
  std::cout<<"start reading transformation_file "<<transformation_file<<"\n";

  // read transformation
  std::ifstream infile(transformation_file);
  std::vector<Eigen::Matrix4f> TFs;  

  if (infile.is_open()) {
    float t11, t12 ,t13, t14, t21, t22, t23, t24, t31, t32, t33, t34;
    while(infile >> t11 >> t12 >> t13 >> t14 >> t21 >> t22 >> t23 >> t24 >> t31 >> t32 >> t33 >> t34) {
      Eigen::Matrix4f TF = Eigen::Matrix4f::Identity();
      TF(0,0) = t11;
      TF(0,1) = t12;
      TF(0,2) = t13;
      TF(0,3) = t14;
      TF(1,0) = t21;
      TF(1,1) = t22;
      TF(1,2) = t23;
      TF(1,3) = t24;
      TF(2,0) = t31;
      TF(2,1) = t32;
      TF(2,2) = t33;
      TF(2,3) = t34;
      TFs.push_back(TF);
      // std::cout<<"TF: \n"<< t11 << t12 << t13 << t14 << t21 << t22 << t23 << t24 << t31 << t32 << t33 << t34<<std::endl;
    }
    if (!infile.eof()) {
    }
    infile.close();
  } else {
      std::cerr<<" transformation file "<<transformation_file<<" not found!\n";
  }
  std::cout<<"the transformation_file has been stored in TFs\n";

  // cvo initialization setup
  kitti.set_start_index(0);
  cvo::CvoGPU cvo_align(cvo_param_file);  

  // start the iteration
  cv::Mat source_left, source_right;
  std::vector<float> semantics_source;
  //kitti.read_next_stereo(source_left, source_right, 19, semantics_source);
  kitti.read_next_stereo(source_left, source_right);
  std::shared_ptr<cvo::RawImage> source_raw(new cvo::RawImage(source_left));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw, source_right, calib));
  Eigen::Matrix4f source_tf = TFs[0];
  std::cout<<"source point cloud has been generated\n";
  
  for (int i = 0; i<total_iters-1 ; i++) {   
    
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Calculating indicator for frame "<<i<<" and "<<i+1<<" with "<<method<<std::endl;

    kitti.next_frame_index();
    cv::Mat left, right;

    vector<float> semantics_target;
    //if (kitti.read_next_stereo(left, right, 19, semantics_target) != 0) {
    if (kitti.read_next_stereo(left, right) != 0) {
      std::cout<<"finish all files\n";
      break;
    }
    std::shared_ptr<cvo::RawImage> target_raw(new cvo::RawImage(left));
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_raw, right, calib));

    // get initial guess from transformation file
    Eigen::Matrix4f target_tf = TFs[i+1];  
    //Eigen::Matrix4f init_guess = target_tf * source_tf.inverse(); // from source frame to the target frame
    Eigen::Matrix4f init_guess = source_tf.inverse() * target_tf; // from source frame to the target frame

    // checking the tf
    std::cout<<"accumed source_tf is \n"<<source_tf<<std::endl;
    std::cout<<"accumed target_tf is \n"<<target_tf<<std::endl;
    std::cout<<"init_guess tf: \n"<<init_guess<<std::endl;

    // get the inverse
    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();

    // compute indicator value and store in file 
    float indicator_value = cvo_align.function_angle(*source, *target, init_guess_inv, false, false);
    std::cout<<"indicator value = "<<indicator_value<<std::endl;

    indicator_file << i+1 << " " << indicator_value<<"\n"<<std::flush;

    std::cout<<"\n\n===========next frame=============\n\n";

    // update source frame and the tf for source point cloud
    source_tf = target_tf;
    source = target;

  }
  indicator_file.close();

  return 0;
}
