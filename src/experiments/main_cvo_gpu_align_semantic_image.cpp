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
#include "cvo/CvoParams.hpp"

using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::KittiHandler kitti(argv[1], 0);
  int total_iters = kitti.get_total_number();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  kitti.set_start_index(start_frame);
  int max_num = std::stoi(argv[5]);
  
  accum_output <<"1 0 0 0 0 1 0 0 0 0 1 0\n";
  
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

  //init_guess(2,3)=0;
 /* 
  // 05 538
  init_guess <<
    0.9982213 ,  0.01526773, -0.0576215, -0.02393033,
    -0.0153228 ,  0.9998828 , -0.00050758, -0.00169955,
    0.05760757,  0.00139014,  0.99833823,  0.33685977,
    0.       ,   0. ,         0.        ,  1.        ;
  */
  /*
  //07: 631
  init_guess <<
    0.99996909 , 0.00252929 , 0.00740704 ,-0.07812081,
    -0.00253296,  0.99999591,  0.00048523, -0.02792174,
    -0.00740572, -0.00050394,  0.9999729 ,  0.59069083,
    0.         , 0.         , 0.         , 1.        ;
 */ 
  /*  
   init_guess << 
   0.99997081, -0.00067641, -0.00768325,  0.0659839, 
 0.00066008  ,0.99999708 ,-0.00211982  ,0.00003192,
  0.00768426 , 0.00211469 , 0.99996779 , 0.99285757,
  0.         , 0.         , 0.         , 1.       ; 
  */
  /*
  // 01: 264
  init_guess <<
   0.99997631, -0.00067165,  0.0068385,  -0.09453087,
  0.00068353 , 0.99999886 ,-0.00174244, -0.00678228,
-0.00683664  ,0.00174704  ,0.99997478 , 2.6214076 ,
  0.         , 0.         , 0.        ,  1.        ;
 */ 
  /*
  init_guess <<
   0.99999533, -0.00285432,  0.00093057, -0.00943301,
  0.00285291 , 0.99999478 , 0.00145044 ,-0.02358547,
 -0.00093543 ,-0.00144773 , 0.99999849 , 2.34234614,
  0.         , 0.         , 0.         , 1.        ;
*/
/*
  // 01: 276
  init_guess << 
    0.99999543,  0.00075004 ,-0.00287568 , 0.11622182,
    -0.0007469   ,0.99999794,  0.00136871, -0.06059335,
    0.0028771  ,-0.00136674 , 0.99999464 , 2.46390501,
    0.          ,0.         , 0.         , 1.        ;
<<<<<<< HEAD

  */
  /*
  // 05 544
  init_guess <<
    0.99872   , -0.00812487,   0.0499338,   0.0290202,
    0.00806004,    0.999966,  0.00149955,   0.0222099,
    -0.0499443, -0.00109517,    0.998751,   -0.377143,
    -0        ,   0        ,  -0        ,   1;
  init_guess = init_guess.inverse().eval();
 */ 

/*
01:1074
  init_guess << 
 0.99950189 , 0.00224    , 0.0314794  , 0.04486028,
 -0.00222771 , 0.99999744 ,-0.00042506, -0.02349048,
 -0.03148026 , 0.00035473 , 0.99950426,  1.68980931,
  0.         , 0.         , 0.        ,  1.        ;
*/
 /*
  // 02: 4352
  init_guess << 
  0.999996 ,-0.00171092, -0.00244798,    0.011453,                                                                                                   
	  0.0017332,    0.999957,  0.00912998,    0.032082,          	  
  0.00243225, -0.00913418,    0.999955  ,  -1.29516 ,                                                                                                          -0   ,        0  ,        -0 ,          1;
  init_guess = init_guess.inverse().eval();
*/
  /*  
// 09: 247
init_guess << 
 0.99979139,  0.00183839,  0.02037217,  0.0007646 ,
-0.00179394,  0.99999554, -0.00220001, -0.02851039,
-0.02037605,  0.00216301,  0.99978968,  1.39735539,
 0.        ,  0.        ,  0.        ,  1.        ;
  */
/*
// 10: 91
init_guess <<
 0.99999664,  0.00236829, -0.00078601, -0.01428545,
-0.00236702,  0.99999561,  0.0015615 , -0.02406281,
 0.00079039, -0.00155975,  0.99999769,  1.02715167,
 0.        ,  0.        ,  0.        ,  1.        ;
*/ 
/*
  // 01: 279
  init_guess <<
   0.99999702,  0.00001815, -0.00230026,  0.06650119,
  -0.00001762,  0.99999971,  0.00023573, -0.02922668,
   0.00230013, -0.00023566,  0.99999791,  2.57250444,
   0.        ,  0.        ,  0.        ,  1.        ;
*/

  /*
  // 01: 318
  init_guess <<
    0.99999906, -0.00064654, -0.000879  , -0.01320788,
    0.00064725,  0.99999957,  0.00077941, -0.02785721,
    0.0008785 , -0.00078   ,  0.99999959,  2.64095405,
    0,0,0,1;
  */
 Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  // start the iteration

  cv::Mat source_left, source_right;
  std::vector<float> semantics_source;
  kitti.read_next_stereo(source_left, source_right, NUM_CLASSES, semantics_source);
  //kitti.read_next_stereo(source_left, source_right);
  std::shared_ptr<cvo::RawImage> source_raw(new cvo::RawImage(source_left, NUM_CLASSES, semantics_source));
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw, source_right, calib));
  //source->write_to_color_pcd(std::string(argv[1]) +std::string( "/cvo_semantic_pcds/") + std::string(argv[4]) + std::string(".pcd")) ;
  //source->write_to_txt(std::string(argv[1]) +std::string( "/cvo_points_semantics/") + std::string(argv[4]) + std::string(".txt") );
  double total_time = 0;
  int i = start_frame;
  for (; i<min(total_iters, start_frame+max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;

    kitti.next_frame_index();
    cv::Mat left, right;
    vector<float> semantics_target;
    if (kitti.read_next_stereo(left, right, 19, semantics_target) != 0) {
    //if (kitti.read_next_stereo(left, right) != 0) {
      std::cout<<"finish all files\n";
      break;
    }
    std::shared_ptr<cvo::RawImage> target_raw(new cvo::RawImage(left, NUM_CLASSES, semantics_target));
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_raw, right, calib));

    //if (i == start_frame)
    //  target->write_to_color_pcd("target.pcd");
    //target->write_to_color_pcd(std::string(argv[1]) + std::string("/cvo_semantic_pcds/") + std::to_string(i) + std::string(".pcd") );
    
    Eigen::Matrix4f result, init_guess_inv;
    Eigen::Matrix4f identity_init = Eigen::Matrix4f::Identity(); 
    
    //if (i==start_frame) {
    //  cvo::CvoPointCloud prev_target;
    //  std::cout<<"init is "<<init_guess<<std::endl;
    //  cvo::CvoPointCloud::transform(init_guess, *target, prev_target);
    //  prev_target.write_to_color_pcd("prev_target.pcd");
    // }
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());

    std::cout<<std::flush;
    double time_curr = 0;
    cvo_align.align(*source, *target, init_guess_inv, result,nullptr, &time_curr);
    total_time += time_curr;
    
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
    if (i==start_frame) {
      cvo::CvoPointCloud t_target;
      cvo::CvoPointCloud::transform(result, *target, t_target);
      t_target.write_to_color_pcd("t_target.pcd");
    }    
    // log accumulated pose

    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;
    std::cout<<"\n\n===========next frame=============\n\n";

    source = target;
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_decay_rate = ell_decay_rate;
      init_param.ell_decay_start = ell_decay_start;
      cvo_align.write_params(&init_param);
    }

  }

  std::cout<<"Ave alignment (excluding IO) time per frame is "<<total_time / (double)(i - start_frame + 1)<<std::endl;
  accum_output.close();

  return 0;
}
