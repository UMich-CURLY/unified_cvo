#pragma once

//#include "cupointcloud/cupointcloud.h"
//#include "cukdtree/cukdtree.h"
#include "utils/CvoPoint.hpp"

namespace perl_registration {
  template <typename T> class cuKdTree;
  template <typename T> class cuPointCloud;
  
}



namespace cvo {
  // Cuda Specific Types
  using CuKdTree = perl_registration::cuKdTree<CvoPoint>;
  using CvoPointCloudGPU =  perl_registration::cuPointCloud<CvoPoint>;
}
