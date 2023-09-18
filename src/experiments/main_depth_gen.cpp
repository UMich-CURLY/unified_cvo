#include <algorithm>
#include <iostream>
#include "utils/StaticStereo.hpp"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include "dataset_handler/FinnForestHandler.hpp"
#include "dataset_handler/DataHandler.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/def_assert.hpp"
//#include "utils/CvoPoint.hpp"
//#include "pcl/point_cloud.hpp"
#include "utils/CvoPointCloud.hpp"
//#include "cvo/CvoGPU.hpp"
//#include "cvo/CvoParams.hpp"
#include "cnpy.h"

using namespace std;
namespace fs = boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::unique_ptr<cvo::DatasetHandler> dataset;
  std::string dataset_name(argv[1]);
  std::string dataset_path(argv[2]);
  if (std::strcmp(dataset_name.c_str(), "kitti") == 0) {
    dataset.reset(new cvo::KittiHandler(dataset_path, cvo::KittiHandler::DataType::STEREO));
  } else if (std::strcmp(dataset_name.c_str(), "finnforest") == 0) {
    dataset.reset(new cvo::FinnForestHandler(dataset_path));
  } else if (std::strcmp(dataset_name.c_str(), "tartanair") == 0) {
    dataset.reset(new cvo::TartanAirHandler(dataset_path));
  } else {
    ASSERT(false, "unknown_dataset "+dataset_name);
  }
    //cvo::DatasetHandler dataset(argv[1], cvo::DatasetHandler::DataType::STEREO);
  int total_iters = dataset->get_total_number();
  string calib_file;
  calib_file = string(argv[2] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  std::string depth_folder(argv[3]);
  std::string pcd_folder(argv[4]);
  int start_frame = std::stoi(argv[5]);
  dataset->set_start_index(start_frame);
  int max_num = std::stoi(argv[6]);

  
  if (!fs::is_directory(depth_folder) || !fs::exists(depth_folder)) { // Check if src folder exists
    fs::create_directory(depth_folder); // create src folder
  }


  int i = start_frame;
  for (; i<min(total_iters, start_frame+max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Gen depth for frame "<<i<<"\n";


    cv::Mat left, right;
    //vector<float> semantics_target;
    //if (dataset.read_next_stereo(left, right, 19, semantics_target) != 0) {
    if (dataset->read_next_stereo(left, right) != 0) {
      std::cout<<"finish all files\n";
      break;
    }
    std::cout<<"image shape: "<<left.rows <<"x"<<left.cols<<std::endl;

    std::shared_ptr<cvo::ImageStereo> target_raw(new cvo::ImageStereo(left, right));
    
    //cv::Mat depth_img_32FC1(left.rows, left.cols, CV_32FC1);
    std::vector<float> depth_img(left.rows * left.cols, 0.0);
    cvo::CvoPointCloud pc(*target_raw, calib, cvo::CvoPointCloud::PointSelectionMethod::FULL);
    std::cout<<"pc size "<<pc.size()<<"\n";

    for (int r = 0; r < left.rows; r++) {
      for (int c = 0; c < left.cols; c++) {
        Eigen::Vector3f xyz;
        cvo::StaticStereo::pt_depth_from_disparity(left.rows, left.cols, c, r,
                                                   target_raw->disparity(),
                                                   calib.intrinsic(),
                                                   calib.baseline(),
                                                   xyz);
        depth_img[r * left.cols + c] = xyz(2);
        //depth_img_32FC1.at<float>(i, j) = xyz(2);
        if (r == 200 && c < 5) std::cout<<"depth_img["<<r*left.cols+c<<"] is "<<xyz(2)<<std::endl;
        
      }
    }
    auto result = std::max_element(depth_img.begin(), depth_img.end());
    std::cout << "max element at: " << std::distance(depth_img.begin(), result) <<"is"<<*result<< '\n';


    std::string curr_frame_id = std::to_string(i);
    auto num_digits = curr_frame_id.size();
    auto num_zeros = 6 - num_digits;
    std::string zero(num_zeros, '0');
    // for (int j = 0; j < num_zeros; j++) zero[j] = '0';
    std::string fpath = depth_folder + "/" + zero + curr_frame_id + ".npy";
    std::cout<<"write to "<<fpath<<std::endl;
    //cv::imwrite();
    cnpy::npy_save(fpath, &depth_img[0], {static_cast<unsigned long>(left.rows*left.cols),
                                          //static_cast<unsigned long>(left.cols)
      }, "w");

    pc.write_to_color_pcd(pcd_folder + "/" + zero + curr_frame_id + ".pcd");
    
    dataset->next_frame_index();
  }

  return 0;

}
