#pragma once
#include <Eigen/Sparse>
#include <vector>
#include <memory>
namespace cvo {

  struct Association {

    using Ptr = std::shared_ptr<Association>;
    
    std::vector<int> source_inliers;
    std::vector<int> target_inliers;
    Eigen::SparseMatrix<float, Eigen::RowMajor> pairs;
  };

}
