#include "utils/CvoFrame.hpp"

namespace cvo {

  struct FullStateIRLS {

    
    
  };

  struct FullStateCPU : FullStateIRLS {
    
  };


  class CvoParams;
  
  struct FullStateGPU : FullStateIRLS {

    std::vector<CvoFrameGPU::Ptr> frames;
    
    std::list<IRLS_State_GPU::Ptr> binary_states;
    
    const CvoParams * params_gpu;
  };
  
}
