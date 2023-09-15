#include "utils/CvoPoint.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/VoxelMap_impl.hpp"

namespace cvo {
    
  template class Voxel<CvoPoint>;
  template class VoxelMap<CvoPoint>;
  template class Voxel<const CvoPoint>;
  template class VoxelMap<const CvoPoint>;
  

}
