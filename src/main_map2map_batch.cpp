#include <map>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include "utils/data_type.hpp"
#include "cvo/Cvo.hpp"
#include "utils/CvoPointCloud.hpp"
using namespace boost::filesystem;
using namespace cvo;

void read_odom_file(const std::string& fname,
                    std::vector<Aff3f, Eigen::aligned_allocator<Aff3f>> & poses) {
  std::ifstream infile(fname);
  if (infile.is_open()) {
    while (infile.eof() == false) {
      Aff3f pose = Aff3f::Identity();
      auto & m = pose.matrix();
      infile >> m(0,0) >> m(0,1) >>m(0,2) >> m(0,3)
             >> m(1,0) >> m(1,1) >>m(1,2) >> m(1,3)
             >> m(2,0) >> m(2,1) >>m(2,2) >> m(2,3);

      poses.push_back(pose);
    }
    infile.close();
  }
  
}

int main (int argc, char ** argv) {

  float init_shift = std::stof(argv[3]);

  std::string odom_file(argv[2]);
  std::vector<Aff3f, Eigen::aligned_allocator<Aff3f>>  poses;
  read_odom_file(odom_file, poses);
  
  std::cout<<"folder: "<<argv[1]<<std::endl<<std::flush;
  std::vector<std::string> files;
  std::vector<std::string> fnames;
  path folder(argv[1]);
  for (auto &p : directory_iterator(folder)) {
    if (is_regular_file(p.path())) {
      std::string curr_file = p.path().string();
      files.push_back(curr_file);
      fnames.push_back(p.path().filename().string() );
    }
  }
  std::cout<<"Read "<<files.size()<<" files \n"<<std::flush;
  std::sort(files.begin(), files.end());
  std::sort(fnames.begin(), fnames.end());
  for (int i = std::stoi(argv[4]); i < files.size() ; i+=2) {
    std::string f1 = fnames[i].substr(0, fnames[i].size()-4);
    std::string f2 = fnames[i+1].substr(0, fnames[i+1].size()-4);

    size_t id1_pos = 1 + f1.find("_");
    size_t id2_pos = 1 + f2.find("_");

    int id1 = std::stoi(f1.substr(id1_pos, f1.size() - id1_pos ));
    int id2 = std::stoi(f2.substr(id2_pos, f2.size() - id1_pos ));

    Aff3f id1_to_id2_true = poses[id1].inverse() * poses[id2];
    id1_to_id2_true.matrix()(2,3) += init_shift;
    Aff3f id1_to_id2 = (poses[id1].inverse() * poses[id2]).inverse();

    CvoPointCloud p1(files[i]);
    CvoPointCloud p2 (files[i+1]);
    if (p1.num_points() < 7000 && p2.num_points() < 7000)
      continue;
    std::cout<<"====================================================\n";
    cvo::cvo cvo_align("cvo_param_map2map_tune.txt");
    std::cout<<" processing frame "<<id1<<" to "<<id2<<",init guess's inner product is \n"<<cvo_align.inner_product(p1, p2, id1_to_id2 .inverse())<<std::endl;
    std::cout<<"Init guess is \n"<<(poses[id1].inverse() * poses[id2]).matrix()<<std::endl;
    cvo_align.set_pcd(p1, p2, id1_to_id2_true.inverse(), true);
    cvo_align.align();
    
    std::cout<<" processing frame "<<id1<<" to "<<id2<<", inner product is "<<cvo_align.inner_product(p1, p2, cvo_align.get_transform())<<std::endl;
    std::cout<<cvo_align.get_transform().matrix()<<std::endl;

    CvoPointCloud target_transformed, target_old;
    CvoPointCloud::transform(cvo_align.get_transform().matrix(), p2, target_transformed);
    CvoPointCloud::transform(id1_to_id2.inverse().matrix(), p2, target_old);
    p1.write_to_color_pcd(std::to_string(i)+"_source.pcd");
    target_transformed.write_to_color_pcd(std::to_string(i)+"_target_new.pcd");
    target_transformed.write_to_color_pcd(std::to_string(i)+"_target_old.pcd");
    break;
  }
  
  return 0;
  
}
