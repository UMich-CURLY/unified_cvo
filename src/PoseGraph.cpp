#include <iostream>
#include <cstdio>

#include "graph_optimizer/PoseGraph.hpp"

namespace cvo {
  PoseGraph::PoseGraph() {
    
    
  }

  PoseGraph::~PoseGraph() {
    
    
  }

  void PoseGraph::add_new_frame(std::shared_ptr<Frame> new_frame) {
    std::cout<<"add new frame id "<<new_frame->id<<std::endl;
  }

  void PoseGraph::optimize() {
    std::cout<<"optimize\n";
    
  }
  
}
