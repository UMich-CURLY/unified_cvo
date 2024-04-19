#pragma once


//#include "cvo/local_parameterization_se3.hpp"
//#include "utils/PointSegmentedDistribution.hpp"
//#include "utils/CvoPointCloud.hpp"
//#include "cvo/IRLS_State.hpp"
//#include "utils/CvoFrame.hpp"
//#include "cvo/CvoParams.hpp"
//#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <list>
#include <vector>
#include <memory>

namespace cvo {

  class CvoFrame;
  //class CvoFrame::Ptr;
  class CvoParams;
  class BinaryState;
  ///class BinaryState::Ptr;

  class CvoBatchIRLS {
  public:

    using Mat34d_row = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
    
    CvoBatchIRLS(const std::vector<std::shared_ptr<CvoFrame>> & frames,
                 const std::vector<bool> & pivot_flags,
                 const std::list<std::shared_ptr<BinaryState>> & states,
                 const CvoParams * params
                 );

    int solve();
    int solve(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & gt,
               const std::string & err_file_name);

  private:
    const std::list<std::shared_ptr<BinaryState>> * states_;
    
    const std::vector<std::shared_ptr<CvoFrame>> * frames_;
    //const FullStates
    //CvoFrame * const pivot_;
    std::vector<bool> pivot_flags_; 
    const CvoParams * params_;


  };

  
  

}
