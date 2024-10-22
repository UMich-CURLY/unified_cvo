#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <filesystem>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/CvoParams.hpp"
#include "utils/CvoPoint.hpp"
#include "utils/PointSegmentedDistribution.hpp"

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::string in_pcd_file(argv[1]);
  std::string out_pcd_file(argv[2]);

  if ( !std::filesystem::exists( in_pcd_file ) ) {
    std::cout<< in_pcd_file <<" doesn't exist!\n";
    return 0;
  }
  

  using PCLType = CvoPointToPCL<cvo::CvoPoint>::type;
  pcl::PointCloud<cvo::CvoPoint>::Ptr cloud (new pcl::PointCloud<cvo::CvoPoint>);
  pcl::io::loadPCDFile<cvo::CvoPoint>(in_pcd_file, *cloud);

  pcl::PointCloud<PCLType>::Ptr result (new pcl::PointCloud<PCLType>);  
  pcl::PointSeg_to_PCL<cvo::CvoPoint::FEATURE_DIMENSION,
                       cvo::CvoPoint::LABEL_DIMENSION
                       >(*cloud, *result);

  pcl::io::savePCDFileASCII(out_pcd_file, *result);

  return 0;
}
