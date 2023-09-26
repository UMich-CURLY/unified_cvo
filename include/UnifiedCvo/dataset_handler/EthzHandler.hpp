#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "DataHandler.hpp"

namespace cvo
{

    class EthzHandler : public DatasetHandler
    {
    public:
        enum FrameType
        {
            LOCAL = 0,
            GLOBAL
        };
        EthzHandler(std::string dataset_folder, FrameType frame_type);
        ~EthzHandler();
        int read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc);
        void read_ground_truth_poses(std::vector<double> &timestamp, std::vector<Eigen::Matrix4d> &poses);
        void next_frame_index();
        void set_start_index(int start);
        int get_current_index();
        int get_total_number();

    private:
        int curr_index;
        std::vector<std::string> names;
        std::string dataset_folder;
        std::string data_folder;
        std::ifstream infile;

        bool debug_plot;
    };

}
