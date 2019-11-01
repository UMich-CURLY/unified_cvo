#include "utils/StaticStereo.hpp"

namespace cvo{
  static StaticStereo::TraceStatus StaticStereo::trace_stereo(const cv::Mat & left,
                                                              const cv::Mat & right,
                                                              const Mat33f & intrinsic,
                                                              const float baseline, // left->right > 0
                                                              pair<float, float> & result
                                                              ) const  {
    Vec3f bl;
    bl << baseline, 0, 0;
    
  }
  
}
