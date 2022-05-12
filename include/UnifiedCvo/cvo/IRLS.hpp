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

    void solve();

  private:
    const std::list<std::shared_ptr<BinaryState>> * states_;
    
    const std::vector<std::shared_ptr<CvoFrame>> * frames_;
    //const FullStates
    //CvoFrame * const pivot_;
    const std::vector<bool> * pivot_flags_; 
    const CvoParams * params_;

  };

  
  

}
