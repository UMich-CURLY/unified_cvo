#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"

using namespace std;
namespace fs = boost::filesystem;


std::vector<std::string> get_all(fs::path const & root, std::string const & ext) {
  std::vector<std::string> paths;

  if (fs::exists(root) && fs::is_directory(root))
  {
    for (auto const & entry : fs::recursive_directory_iterator(root))
    {
      if (fs::is_regular_file(entry) && entry.path().extension() == ext)
        paths.emplace_back(entry.path().string());
    }
  }
  return paths;
}  

int main(int argc, char** argv) {

  std::string kitti_folder(argv[1]);
  std::string cvo_param_file(argv[2]);
  std::string init_guess_folder(argv[3]);
  int is_using_semantic = std::stoi(std::string(argv[4]));
  cvo::CvoGPU cvo_align(cvo_param_file);
      
  std::vector<std::string> paths =  get_all(init_guess_folder, ".txt");
  for (auto && file_path : paths ) {
    std::string seq_name, frame_name;
    std::ifstream infile(file_path);
    infile >> seq_name >> frame_name;
    int seq_id = std::stoi(seq_name);
    int frame_id = std::stoi(frame_name);
    std::cout<<"\n\n\n\n=======================================\n New seq "<<seq_id<<" frame "<<frame_id<<" and "<<frame_id+1<<std::endl;
    string calib_file;
    calib_file = string(argv[1] ) + "/" + seq_name +"/cvo_calib.txt"; 
    cvo::Calibration calib(calib_file);


    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    infile >> init_guess(0,0) >> init_guess(0,1) >> init_guess(0,2) >> init_guess(0,3)
           >> init_guess(1,0) >> init_guess(1,1) >> init_guess(1,2) >> init_guess(1,3)
           >> init_guess(2,0) >> init_guess(2,1) >> init_guess(2,2) >> init_guess(2,3);

    infile.close();
    
    cvo::KittiHandler kitti(kitti_folder + "/" + seq_name, 0);
    kitti.set_start_index(frame_id);
    
    cv::Mat source_left, source_right;
    std::vector<float> semantics_source;
    if (is_using_semantic)
      kitti.read_next_stereo(source_left, source_right, NUM_CLASSES, semantics_source);
    else
      kitti.read_next_stereo(source_left, source_right);
    
    std::shared_ptr<cvo::RawImage> source_raw;

    if (is_using_semantic)
      source_raw.reset(new cvo::RawImage(source_left, NUM_CLASSES, semantics_source));
    else
      source_raw.reset(new cvo::RawImage(source_left));
    std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw, source_right, calib));
    source->write_to_color_pcd(seq_name + "_" + frame_name + ".pcd");
    kitti.next_frame_index();
    cv::Mat target_left, target_right;
    std::vector<float> semantics_target;
    if (is_using_semantic)
      kitti.read_next_stereo(target_left, target_right, NUM_CLASSES, semantics_target);
    else
      kitti.read_next_stereo(target_left, target_right);
    
    std::shared_ptr<cvo::RawImage> target_raw;

    if (is_using_semantic)
      target_raw.reset(new cvo::RawImage(target_left, NUM_CLASSES, semantics_target));
    else
      target_raw.reset(new cvo::RawImage(target_left));
    std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_raw, target_right, calib));
    target->write_to_color_pcd(seq_name + "_" + std::to_string(frame_id+1) + ".pcd");
    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());

    std::cout<<std::flush;
    cvo_align.align(*source, *target, init_guess_inv, result);
    
    std::cout<<"Cvo align result for seq "<<seq_name<<" frame id "<<frame_name<<" is \n"<<result<<std::endl;
    
    
  }
  return 0;
}
