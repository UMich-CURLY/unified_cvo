#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
  if (argc < 3){
    std::cerr<<"Needs two pointclouds input\n";
    return -1;
  }

  std::string source_name (argv[1]);
  std::string target_name(argv[2]);

  Eigen::Matrix4f init_guess;
  init_guess <<
   0.99997631, -0.00067165,  0.0068385,  -0.09453087,
  0.00068353 , 0.99999886 ,-0.00174244, -0.00678228,
-0.00683664  ,0.00174704  ,0.99997478 , 2.6214076 ,
  0.         , 0.         , 0.        ,  1.        ;


  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>);
 pcl::PointCloud<pcl::PointXYZRGB>::Ptr   target(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr  final_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(source_name, *source);
  pcl::io::loadPCDFile(target_name, *target);
  std::cout<<"source and target size is "<<source->size()<<", "<<target->size()<<std::endl;

  pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> pcl_ndt;
  Eigen::Matrix4f ndt_result;
  pcl_ndt.setTransformationEpsilon(1e-6);
    //pcl_ndt.setStepSize(0.1);
    //pcl_ndt.setResolution(2.0);
  pcl_ndt.setMaximumIterations(150);
  pcl_ndt.setInputSource( source );
  pcl_ndt.setInputTarget( target );
  pcl_ndt.align(*final_cloud, init_guess );
  std::cout<<"NDT result is \n"<<pcl_ndt.getFinalTransformation()<<std::endl;
  

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> pcl_gicp;
  pcl_gicp.setInputSource (source);
  pcl_gicp.setInputTarget (target);
  pcl_gicp.setMaximumIterations (200);
  //pcl_gicp.setTransformationEpsilon (1e-8);
  pcl_gicp.align (*final_cloud , init_guess )  ;
  std::cout<<"GICP result is \n"<<pcl_gicp.getFinalTransformation()<<std::endl;
                                       
  return 0;
  
  
}


  
