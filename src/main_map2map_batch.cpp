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

  cvo::cvo cvo_align; 

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
  for (int i =0; i < files.size() ; i+=2) {
    std::string f1 = fnames[i].substr(0, fnames[i].size()-4);
    std::string f2 = fnames[i+1].substr(0, fnames[i+1].size()-4);

    size_t id1_pos = 1 + f1.find("_");
    size_t id2_pos = 1 + f2.find("_");

    int id1 = std::stoi(f1.substr(id1_pos, f1.size() - id1_pos ));
    int id2 = std::stoi(f2.substr(id2_pos, f2.size() - id1_pos ));

    Aff3f id1_to_id2 = (poses[id1].inverse() * poses[id2]).inverse();

    CvoPointCloud p1(files[i]);
    CvoPointCloud p2 (files[i+1]);

    std::cout<<" processing frame "<<id1<<" to "<<id2<<",init guess's inner product is \n"<<cvo_align.inner_product(p1, p2, id1_to_id2.inverse())<<std::endl;

    cvo_align.set_pcd(p1, p2, id1_to_id2, true);
    cvo_align.align();
    
    std::cout<<" processing frame "<<id1<<" to "<<id2<<", inner product is "<<cvo_align.inner_product()<<std::endl;
    std::cout<<cvo_align.get_transform().matrix()<<std::endl;
  }
  
  return 0;
  
}
