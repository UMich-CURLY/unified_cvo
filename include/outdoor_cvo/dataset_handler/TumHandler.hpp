#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "DataHandler.hpp"


namespace cvo{

  class TumHandler : public DatasetHandler{
		public:
			TumHandler(std::string tum_folder);
			int read_next_rgbd(cv::Mat & rgb_img, 
                        cv::Mat & dep_img);

			void next_frame_index();
			void set_start_index(int start);
			int get_current_index();
			int get_total_number();
			std::vector<std::string> get_rgb_name_list(){return vstrRGBName;};
			
		private:
			/**
			 *  @brief load the association file and get paired names of rgb and depth images.
			**/
			void load_file_name(std::string assoc_pth, std::vector<std::string> &vstrRGBName, \
                    std::vector<std::string> &vstrRGBPth, std::vector<std::string> &vstrDepPth);

			int curr_index;
			std::vector<std::string> names;
			std::vector<std::string> vstrRGBName;     // vector for image names
			std::vector<std::string> vstrRGBPth;
			std::vector<std::string> vstrDepPth;
			std::string folder_name;
			std::ifstream infile;

			bool debug_plot;
  };

}