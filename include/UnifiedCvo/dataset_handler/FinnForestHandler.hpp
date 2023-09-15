#pragma once
#include "DataHandler.hpp"
#include <vector>
#include <string>


namespace cvo {
  class FinnForestHandler : public DatasetHandler {
  public:
    // data_type: 0: Stereo,  1: Lidar
    FinnForestHandler(const std::string & folder);
    ~FinnForestHandler();
    int read_next_stereo(cv::Mat & left,
                         cv::Mat & right);
    int read_next_stereo(cv::Mat & left,
                         cv::Mat & right,
                         int num_semantic_class,
                         std::vector<float> & left_semantics);

    //std::map<int,int> create_label_map();
    
    void next_frame_index();
    void set_start_index(int start);
    int get_current_index();
    int get_total_number();
  private:

    int curr_index;
    std::vector<std::string> names;
    std::string folder_name;

    const std::string left_cam = "images_cam2_sr22555667";
    const std::string right_cam = "images_cam2_sr22555667";

    bool debug_plot;

  };
}
