#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include "DataHandler.hpp"


namespace cvo{

  class TartanAirHandler : public DatasetHandler{
  public:
    TartanAirHandler(std::string tartan_traj_folder);
    ~TartanAirHandler();
    void set_depth_folder_name(const std::string & folder);
    int read_next_rgbd(cv::Mat & rgb_img, 
                       cv::Mat & dep_img);
    int read_next_rgbd(cv::Mat & rgb_img,
                       std::vector<float> & dep_vec);
    int read_next_rgbd(cv::Mat & rgb_img,
                       std::vector<float> & dep_vec,
                       int num_semantic_class,
                       std::vector<float> & semantics);

    int read_next_rgbd_wihtout_sky(cv::Mat & rgb_img,
                                   std::vector<float> & dep_vec,
                                   int num_semantic_class,
                                   std::vector<float> & semantics,
                                   int sky_label);
    
    int read_next_rgbd_with_flow(cv::Mat & rgb_img,
                                 std::vector<float> & depth,
                                 std::vector<float> & flow_curr_to_next);
    int read_next_flow(std::vector<float> & flow_curr_to_next);
    
    int read_next_stereo(cv::Mat & left,
                         cv::Mat & right);
    int read_next_stereo(cv::Mat & left,
                         cv::Mat & right,
                         int num_semantic_class,
                         std::vector<float> & semantics);
    void next_frame_index();
    void set_start_index(int start);
    int get_current_index();
    int get_total_number();
			
  private:
    int read_next_semantics(int num_pixels,
                            int num_semantic_class,
                            std::vector<float> & semantics);

  private:
    int curr_index;
    int total_size;
    std::string folder_name;
    std::unordered_map<uint8_t, uint8_t> semantic_class;
    std::string depth_folder_name;

  };

}
