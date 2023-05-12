#pragma once

#include "cupointcloud/cupointcloud.h"
#include "cukdtree/cukdtree.h"
#include "utils/CvoPoint.hpp"

namespace perl_registration {
  // Cuda Specific Types
  template class cuKdTree<cvo::CvoPoint>;
  template class cuPointCloud<cvo::CvoPoint>;
  

}
