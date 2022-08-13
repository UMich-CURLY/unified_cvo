#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

namespace cvo {
  template <typename Dtype, unsigned int Major>
  std::string mat_to_line(const Eigen::Matrix<Dtype, 4, 4, Major> & mat) {
    std::string ret;
    for (int i = 0; i < 16; i++) {
      int col = i % 4;
      int row = i / 4;
      if (i == 15)
        ret = ret + std::to_string(mat(row, col));
      else
        ret = ret + std::to_string( mat(row, col)) + " ";        
    }
    return ret;
  }
}
