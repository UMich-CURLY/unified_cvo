#pragma once
#include <Eigen/Sparse>
#include <vector>

namespace cvo {

  struct Association {
    std::vector<int> source_inliers;
    std::vector<int> target_inliers;
    Eigen::SparseMatrix<float, Eigen::RowMajor> pairs;
  };

}
