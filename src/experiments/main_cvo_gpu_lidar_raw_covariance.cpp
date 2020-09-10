#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
#include "dataset_handler/KittiHandler.hpp"
//#include "graph_optimizer/Frame.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
//#include "cvo/Cvo.hpp"
using namespace std;
using namespace boost::filesystem;


pcl::visualization::PCLVisualizer::Ptr covariance_visualizer (const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES> > & cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointSeg_to_PointXYZ(cloud, *cloud_xyz);

  std::cout<<"construct new viewer...\n"<<std::flush;
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("PointCloud Covariance Viewer"));

  std::cout<<" add points and  configs to viewer\n"<<std::flush;
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud_xyz, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

  std::cout<<"add covarainces to viewer\n"<<std::flush;
  for (int i = 0; i < cloud_xyz->size(); i++) {
    float radius = cloud[i].cov_eigenvalues[2] ;
    if (radius > 2) radius = 2;
    if (radius < 0.05) radius = 0.05;
    
    if (i < 100) std::cout<<"sphere radius is "<<radius<<std::endl;
    viewer->addSphere (cloud_xyz->points[i] ,  radius , 20, 20, 10, "sphere"+std::to_string(i));      
  }
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  std::cout<<" return\n"<<std::flush;
  return (viewer);
}  



void convert_to_pcl(const cvo::CvoPointCloud & cvo_cloud,
                    pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES> > & pcl_cloud) {
  int num_points = cvo_cloud.num_points();
  auto & positions = cvo_cloud.positions();
  const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
  //auto & features = this->features_;
  auto & labels = cvo_cloud.labels();
  auto & covariance = cvo_cloud.covariance();
  auto & eigenvalues = cvo_cloud.eigenvalues();
    
  pcl_cloud.points.resize(num_points);
  pcl_cloud.width = num_points;
  pcl_cloud.height = 1;
  std::cout<<"start converting to pcl\n";
  for(int i=0; i<num_points; ++i){
    // set positions
    pcl_cloud.points[i].x = positions[i](0);
    pcl_cloud.points[i].y = positions[i](1);
    pcl_cloud.points[i].z = positions[i](2);
    if (FEATURE_DIMENSIONS > 1) {
      pcl_cloud.points[i].r = (uint8_t)std::min(255.0, (features(i,0) * 255.0));
      pcl_cloud.points[i].g = (uint8_t)std::min(255.0, (features(i,1) * 255.0)) ;
      pcl_cloud.points[i].b = (uint8_t)std::min(255.0, (features(i,2) * 255.0));
    }
    
    for (int j = 0; j < FEATURE_DIMENSIONS; j++)
      pcl_cloud[i].features[j] = features(i,j);

    memcpy(pcl_cloud.points[i].covariance, covariance.data() + i*9, sizeof(float)*9);
    memcpy(pcl_cloud.points[i].cov_eigenvalues, eigenvalues.data() +i*3, sizeof(float)*3);
    if (i < 3 )std::cout<<"copied pcl cov is "<<pcl_cloud.points[i].cov_eigenvalues[0]<<", "<<pcl_cloud.points[i].cov_eigenvalues[2]<<std::endl;
  }

  std::cout<<" Finish converting to pcl\n";
  
    
}


int main(int argc, char *argv[]) {
  std::cout<<"start\n";
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::KittiHandler kitti(argv[1], 1);
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

  std::cout<<"new cvo_align\n";
  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  float ell_init = init_param.ell_init;
  float ell_max = init_param.ell_max;
  init_param.ell_init = init_param.ell_init_first_frame;
  init_param.ell_max = init_param.ell_max;
  cvo_align.write_params(&init_param);
  
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3) = 0;
  
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  // start the iteration

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
  kitti.read_next_lidar(source_pc);
  std::cout<<"[main]read next lidar\n"; 
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(source_pc, 64));
  //cvo::CvoPointCloud * source = new cvo::CvoPointCloud(source_pc, 64);
  std::cout<<"[main] read complete\n"<<std::flush;

  //write_to_pcl(cvo_cloud, "lidar_pcl.pcd");
  pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES>> pcl_cloud;
  convert_to_pcl(*source, pcl_cloud);
  //source->write_to_intensity_pcd("kitti_pcl/"+std::to_string(start_frame)+".pcd" );
  pcl::io::savePCDFileASCII<pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES>>("kitti_pcl/"+std::to_string(start_frame)+".pcd" ,pcl_cloud);
  
  if (init_param.is_pcl_visualization_on == 1) {
    std::cout<<"converted to pcl!\n";
    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer =  covariance_visualizer(pcl_cloud);
    while (!viewer->wasStopped ()) {
      viewer->spinOnce (100);
      std::this_thread::sleep_for(100ms);
    }
  }



  double total_time = 0;
  int i = start_frame;
  for (; i<min(total_iters, start_frame+max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;
    
    // pcl::PointCloud<pcl::PointXYZI>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZI>);
    //kitti.read_next_lidar(source_pc);
    //std::cout<<"[main]read next lidar\n"; 
    //std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(source_pc, 64));
    //cvo::CvoPointCloud source(source_pc, 64);
    //std::cout<<"[main] read complete\n"<<std::flush;

    kitti.next_frame_index();
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZI>);
    if (kitti.read_next_lidar(target_pc) != 0) {
      std::cout<<"finish all files\n";
      break;
    }
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(target_pc, 64));
    //cvo::CvoPointCloud * target = new cvo::CvoPointCloud(target_pc, 64);
    //cvo::CvoPointCloud target (target_pc, 64);
    //pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES>> pcl_target;
    //convert_to_pcl(*target, pcl_target);
    //pcl::io::savePCDFileASCII<pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS,NUM_CLASSES>>("kitti_pcl/"+std::to_string(i+1)+".pcd" ,pcl_target);  
    //target->write_to_intensity_pcd("kitti_pcl/"+std::to_string(i+1)+".pcd"); 
    std::cout<<"NUm of source pts is "<<source->num_points()<<"\n";
    std::cout<<"NUm of target pts is "<<target->num_points()<<"\n";

    
    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
    std::cout<<std::flush;
    double this_time = 0;
    cvo_align.align(*source, *target, init_guess_inv, result, &this_time);
    total_time += this_time;
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product(*source, *target, result);

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
    //delete source;
    std::cout<<"just swtich source and target\n"<<std::flush;
    source = target;
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
