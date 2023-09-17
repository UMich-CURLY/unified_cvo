#pragma once
//#include "utils/PointSegmentedDistribution.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
//#include "utils/CvoPointCloud.hpp"

namespace cvo {

  class CvoPointCloud;
  
  struct CvoFrame {
  public:

    typedef std::shared_ptr<CvoFrame> Ptr;
    
    CvoFrame(const CvoPointCloud * pts,
             const double poses[12],
             bool is_using_kdtree);

    virtual ~CvoFrame() {}

    const CvoPointCloud * points; // no ownership
    
    
    double pose_vec[12]; // 3x4 row-order matrix. [R t]
    Eigen::Matrix4d pose_cpu() const;
    
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > & points_transformed();
    virtual void transform_pointcloud();

    //virtual unsigned int get_id() const { return this; }
    
  private:

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > points_transformed_;    
    
    
  };
}
