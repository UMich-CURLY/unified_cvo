#pragma once
#include <Eigen/Sparse>
#include <vector>

namespace cvo {

  struct Association {
    std::vector<bool> source_inliers;
    std::vector<bool> target_inliers;
    Eigen::SparseMatrix<float, Eigen::RowMajor> pairs;
  };

}
