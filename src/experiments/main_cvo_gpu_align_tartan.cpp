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
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  tartan.set_start_index(start_frame);
  int max_num = std::stoi(argv[5]);
  
  
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

  cv::Mat source_rgb;
  vector<float> source_depth;
  tartan.read_next_rgbd(source_rgb, source_depth);
  std::shared_ptr<cvo::ImageRGBD<float>> source_raw(new cvo::ImageRGBD<float>(source_rgb, source_depth));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw,
                                                                    calib
                                                                    ));
  //19, semantics_source, 
  //                                                                    cvo::CvoPointCloud::CV_FAST));
  
  for (int i = start_frame; i<min(total_iters, max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;

    tartan.next_frame_index();
    cv::Mat rgb;
    std::vector<float> dep;    
    //sdt::vector<float> semantics_target;
    if (tartan.read_next_rgbd(rgb, dep) != 0) {
      std::cout<<"finish all files\n";
      break;
    }

    //std::shared_ptr<cvo::Frame> target(new cvo::Frame(i+1, rgb, dep, calib,1));
    std::shared_ptr<cvo::ImageRGBD<float>> target_raw(new cvo::ImageRGBD(rgb, dep));
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_raw, calib
                                                                      //,cvo::CvoPointCloud::CANNY_EDGES
                                                                      ));
    //if (i == 0)
    //  target->write_to_color_pcd("target.pcd");  
    // std::cout<<"reading "<<files[cur_kf]<<std::endl;

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
    std::cout<<std::flush;
    cvo_align.align(*source, *target, init_guess_inv, result);
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product_cpu(*source, *target, result, ell_init);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;
    
    
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

    std::cout<<"\n\n===========next frame=============\n\n";
   
    source = target;
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_decay_rate = ell_decay_rate;
      init_param.ell_decay_start = ell_decay_start;
      
      cvo_align.write_params(&init_param);
      
    }


  }


  accum_output.close();

  return 0;
}
