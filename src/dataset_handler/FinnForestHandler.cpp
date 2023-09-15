#include "dataset_handler/FinnForestHandler.hpp"
#include <algorithm>
#include <boost/filesystem.hpp>

using namespace std;
namespace cvo {
  FinnForestHandler::FinnForestHandler(const std::string & folder ) {
    
    curr_index = 0;
    folder_name = folder;
    debug_plot = true;
    
    std::string data_folder = folder + "/" + left_cam;
    boost::filesystem::path dir(data_folder.c_str());
    for (auto & p : boost::filesystem::directory_iterator(dir)) {
      if (is_regular_file(p.path())) {
        string curr_file = p.path().filename().string();
        size_t last_ind = curr_file.find_last_of(".");
        string raw_name = curr_file.substr(0, last_ind);
        names.push_back(raw_name);
      }
    }
    sort(names.begin(), names.end());
    cout<<"Kitti contains "<<names.size()<<" files\n";
    
  }

  FinnForestHandler::~FinnForestHandler() {
    
  }

  int FinnForestHandler::read_next_stereo(cv::Mat & left,
                                          cv::Mat & right) {
    
    if (curr_index >= names.size())
      return -1;

    string left_name = folder_name + "/" + left_cam + "/" + names[curr_index] + ".png";
    string right_name = folder_name + "/" + right_cam + "/" + names[curr_index] + ".png";
    left = cv::imread(left_name, cv::ImreadModes::IMREAD_COLOR );
    right = cv::imread(right_name, cv::ImreadModes::IMREAD_COLOR );

    if (left.data == nullptr || right.data == nullptr) {
      cerr<<"Image doesn't read successfully: "<<left_name<<", "<<right_name<<"\n";
      return -1;
    }
    return 0;
  }


  void FinnForestHandler::next_frame_index() {
    curr_index ++;
  }


  void FinnForestHandler::set_start_index(int start) {
    curr_index = start;
    
  }

  int FinnForestHandler::get_current_index() {
    return curr_index;
    
  }

  int FinnForestHandler::get_total_number() {
    return names.size();
    
  }
  
  
}
