#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include "dataset_handler/TumHandler.hpp"
#include "utils/debug_visualization.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <map>
using namespace std;
using namespace boost::filesystem;

namespace cvo {
	TumHandler::TumHandler(std::string tum_folder){
		curr_index = 0;
		folder_name = tum_folder;
		debug_plot = true;
		
		// load associate file
		string assoc_pth = tum_folder + "/assoc.txt";
		std::cout<<"assoc path: "<<assoc_pth<<std::endl;
		load_file_name(assoc_pth, vstrRGBName, vstrRGBPth, vstrDepPth);
		
		cout<<"Tum contains "<<vstrRGBName.size()<<" files\n";
	}

	int TumHandler::read_next_rgbd(cv::Mat & rgb_img, 
                        cv::Mat & dep_img){
		
    if (curr_index >= vstrRGBName.size()){
			cerr<<"Can't read images. Image index out of range!"<<"\n";
      return -1;
		}

		string rgb_pth = folder_name + "/" + vstrRGBPth[curr_index];
		string dep_pth = folder_name + "/" + vstrDepPth[curr_index];

		rgb_img = cv::imread(rgb_pth);
    dep_img = cv::imread(dep_pth,CV_LOAD_IMAGE_ANYDEPTH);
		if (rgb_img.data == nullptr || dep_img.data == nullptr) {
      cerr<<"Image doesn't read successfully: "<<rgb_pth<<", "<<dep_pth<<"\n";
      return -1;
    }
    return 0;

	}

	void TumHandler::next_frame_index() {
    curr_index ++;
  }


  void TumHandler::set_start_index(int start) {
    curr_index = start;
  }

  int TumHandler::get_current_index() {
    return curr_index;
  }

  int TumHandler::get_total_number() {
    return vstrRGBName.size();
    
  }

	void TumHandler::load_file_name(string assoc_pth, vector<string> &vstrRGBName, \
                    					vector<string> &vstrRGBPth, vector<string> &vstrDepPth){
    std::ifstream fAssociation;
    fAssociation.open(assoc_pth.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string RGB;
            ss >> RGB;
            vstrRGBName.push_back(RGB);
            string RGB_pth;
            ss >> RGB_pth;
            vstrRGBPth.push_back(RGB_pth);
            string dep;
            ss >> dep;
            string depPth;
            ss >> depPth;
            vstrDepPth.push_back(depPth);
        }
    }
    fAssociation.close();
}


}


