#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "utils/LidarPointType.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
using namespace std;
using namespace boost::filesystem;


template <typename PointT>
void convert_pointcloud_to_pointxyzi(const pcl::PointCloud<PointT> & custom_p,
                                     pcl::PointCloud<pcl::PointXYZI> & out ) {
  out.width = custom_p.width;
  out.height = custom_p.height;
  out.resize(custom_p.size());
  for (int i = 0; i < custom_p.size(); i++) {
    out[i].x = - custom_p[i].y;
    out[i].y = - custom_p[i].z;
    out[i].z = custom_p[i].x;
    out[i].intensity = custom_p[i].intensity;    
  }
  
}

void path_to_filenames(string & folder, vector<string> & filenames) {
  path p(folder);
  if(is_directory(p))
  {
    for (directory_iterator itr(p); itr!=directory_iterator(); ++itr)
    {
      //cout << itr->path().filename() << ' '; // display filename only
      if (is_regular_file(itr->status())) {
        filenames.push_back(itr->path().filename().string());
      }
    }
    sort(filenames.begin(), filenames.end());
  }
}


int main(int argc, char *argv[]) {

  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  string pcd_folder = argv[1];
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);

  int max_num = std::stoi(argv[5]);

  
  vector<string> files;
  path_to_filenames(pcd_folder, files);

  accum_output <<"1 0 0 0 0 1 0 0 0 0 1 0\n";

  std::cout<<"new cvo_align\n";
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_max = init_param.ell_max;
  init_param.ell_init = 0.35;//0.51;
  init_param.ell_max = 1.5;//0.75;
  cvo_align.write_params(&init_param);
  
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0.0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  
  // start the iteration
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
  std::cout<<"reading "<<pcd_folder + "/" + files[start_frame]<<std::endl;
  pcl::PointCloud<pcl::PointXYZIR> source_ir;
  pcl::io::loadPCDFile<pcl::PointXYZIR> (pcd_folder + "/"+  files[start_frame], source_ir );
  convert_pointcloud_to_pointxyzi(source_ir, *source_pc);
  std::shared_ptr<cvo::CvoPointCloud> source_fr (new cvo::CvoPointCloud(source_pc, 5000, 32));

  double total_time = 0;
  int i = start_frame;
  for (; i<min( int(files.size()), start_frame+max_num)-1 ; i=i+5) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;


    pcl::PointCloud<pcl::PointXYZI>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZI>);
    std::cout<<"reading "<<pcd_folder + "/" + files[start_frame+1]<<std::endl;
    pcl::PointCloud<pcl::PointXYZIR> target_ir;
    pcl::io::loadPCDFile<pcl::PointXYZIR> (pcd_folder + "/" + files[start_frame+1], target_ir );
    convert_pointcloud_to_pointxyzi(target_ir, *target_pc);
    std::shared_ptr<cvo::CvoPointCloud> target_fr (new cvo::CvoPointCloud(target_pc, 5000, 32));

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source_fr->num_points(), target_fr->num_points());
    std::cout<<std::flush;
    double this_time = 0;
    cvo_align.align(*source_fr, *target_fr, init_guess_inv, result, &this_time);
    total_time += this_time;
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product(*source_fr, *target_fr, result);

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

    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;
    
    std::cout<<"\n\n===========next frame=============\n\n";
   
    source_fr = target_fr;
    if (i == start_frame) {
      init_param.ell_init = ell_init;
      init_param.ell_max = ell_max;
      cvo_align.write_params(&init_param);
      
    } //else if (i < start_frame + 20)  {
      //init_param.ell_init =  1.0;
      //init_param.ell_max = 1.0;
      //cvo_align.write_params(&init_param);

      
    //}

  }

  std::cout<<"time per frame is "<<total_time / double(i - start_frame + 1)<<std::endl;
  accum_output.close();

  return 0;
}
