#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
#include "dataset_handler/TartanAirHandler.hpp"
//#include "graph_optimizer/Frame.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/CvoParams.hpp"
#include "utils/ImageRGBD.hpp"
using namespace std;
using namespace boost::filesystem;

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::TartanAirHandler tartan(argv[1]);
  int total_iters = tartan.get_total_number();
  //vector<string> vstrRGBName = tum.get_rgb_name_list();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);
  int start_frame = std::stoi(argv[3]);
  tartan.set_start_index(start_frame);
  
  
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

  cv::Mat source_rgb;
  vector<float> source_depth;
  int NUM_CLASS = 19;
  vector<float> source_semantics;
  tartan.read_next_rgbd(source_rgb, source_depth,NUM_CLASS,source_semantics);

 
  std::shared_ptr<cvo::ImageRGBD<float>> source_raw(new cvo::ImageRGBD<float>(source_rgb, source_depth,NUM_CLASS,source_semantics));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw,
                                                                    calib
								    ,cvo::CvoPointCloud::DSO_EDGES
                                                                    )); 
  pcl::PointCloud<cvo::CvoPoint> target_cvo;
  cvo::CvoPointCloud_to_pcl(*source,target_cvo);
  std::cout<<"First point is "<<source->at(0).transpose()<<std::endl;

  //19, semantics_source, 
  //                                                                    cvo::CvoPointCloud::CV_FAST));
  pcl::io::savePCDFileASCII ("/home/bigby/result_cvo" + to_string(start_frame) + ".pcd", target_cvo);
  
  return 0;
}
