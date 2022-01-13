#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "DataHandler.hpp"


namespace cvo{

  class TartanAirHandler : public DatasetHandler{
		public:
			TartanAirHandler(std::string tartan_traj_folder);
            ~TartanAirHandler();
			int read_next_rgbd(cv::Mat & rgb_img, 
                        cv::Mat & dep_img);
			void next_frame_index();
			void set_start_index(int start);
			int get_current_index();
			int get_total_number();
			
		private:
			int curr_index;
            int total_size;
			std::string folder_name;
  };

}
