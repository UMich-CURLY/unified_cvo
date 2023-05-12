// updateCovis Debug use
#include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "utils/VoxelMap.hpp"
#include "utils/VoxelMap_impl.hpp"

namespace cvo {


  //template <>
  //VoxelCoord VoxelMap<SimplePoint>::point_to_voxel_center(const SimplePoint* pt) const;
  template class Voxel<SimplePoint>;
  template class VoxelMap<SimplePoint>;
  template class Voxel<pcl::PointXYZRGB>;
  template class VoxelMap<pcl::PointXYZRGB>;
  template class Voxel<pcl::PointXYZI>;
  template class VoxelMap<pcl::PointXYZI>;
}
