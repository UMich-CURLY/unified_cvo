#include <iostream>
#include <cstdio>

#include "graph_optimizer/PoseGraph.hpp"

namespace cvo {
  PoseGraph::PoseGraph() {
    
    
  }

  PoseGraph::~PoseGraph() {
    
    
  }

  void PoseGraph::add_new_frame(std::shared_ptr<Frame> new_frame) {
    std::cout<<"add_new_frame: id "<<new_frame->id<<std::endl;
    std::cout<<"---- number of points is "<<new_frame->points().num_points()<<std::endl;
    //new_frame->points().write_to_color_pcd(std::to_string(new_frame->id)+".pcd"  );

    /*
    auto & last_keyframe = *all_frames_since_last_keyframe.front();
    auto & last_frame = *all_frames_since_last_keyframe.back();
    
    
    // align with 
    //cvo.set_pcd(last_keyframe.points(), new_frame.points())

    bool is_keyframe = false;


    if(is_keyframe) {
      // cvo align again



      
      
    } else {
      
      
      
    }
    */
  }

  void PoseGraph::pose_graph_optimize() {
    std::cout<<"optimize\n";
    
  }
  
}
