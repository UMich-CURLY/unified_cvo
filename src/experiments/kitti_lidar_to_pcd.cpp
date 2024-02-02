#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "cvo/CvoGPU.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "utils/PoseLoader.hpp"
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/LidarPointDownsampler.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZI>;
extern template class cvo::VoxelMap<pcl::PointXYZI>;


int main(int argc, char ** argv) {
  cvo::KittiHandler kitti(argv[1], cvo::KittiHandler::DataType::LIDAR,
                          cvo::KittiHandler::LidarCamCalibType::LIDAR_FRAME);
  std::string result_pcd_folder(argv[2]);
  float leaf_size = std::stof(argv[3]);
  int is_export_semantic_and_color = std::stoi(argv[4]);

  for (int i = 0; i<= kitti.get_total_number(); i++) {
    
    kitti.set_start_index(i);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> semantics;
    if (-1 == kitti.read_next_lidar(pc_pcl, semantics)) 
      break;

    if (is_export_semantic_and_color) {
    	std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(pc_pcl, 10000, 64 ));
    	std::shared_ptr<cvo::CvoPointCloud> pc = cvo::downsample_lidar_points(false,
        	                                                             pc_pcl,
                	                                                     leaf_size,
                        	                                             semantics
                                	                                     );
   	std::cout<<"After downsampling, # of points for frame "<<i<<" is "<<pc->size()<<"\n";
    	//pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    	pcl::PointCloud<cvo::CvoPoint> pc_curr;
    	pc->export_to_pcd(pc_curr);
    	std::string fname = result_pcd_folder + "/" + std::to_string(i)+".pcd";
    	std::cout<<"fname "<<fname<<"\n";
    	pcl::io::savePCDFileASCII(fname, pc_curr);
    } else {
    	std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(pc_pcl, 10000, 64 ));
    	std::shared_ptr<cvo::CvoPointCloud> pc = cvo::downsample_lidar_points(false,
        	                                                             pc_pcl,
                	                                                     leaf_size,
                        	                                             semantics
                                	                                     );
    	std::string fname = result_pcd_folder + "/" + std::to_string(i)+".pcd";
	pc->write_to_intensity_pcd(fname);
    	std::cout<<" just write "<<pc->size()<<" points to "<<fname<<"\n";
    }
  }
  return 0;
  
}
