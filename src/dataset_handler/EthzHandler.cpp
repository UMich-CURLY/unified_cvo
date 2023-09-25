#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cassert>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include "dataset_handler/EthzHandler.hpp"
#include "utils/debug_visualization.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <map>
using namespace std;
using namespace boost::filesystem;

namespace cvo
{
    EthzHandler::EthzHandler(std::string dataset_folder, FrameType frame_type)
    {
        curr_index = 0;
        string name_prefix;
        this->dataset_folder = dataset_folder;
        if (frame_type == FrameType::LOCAL)
        {
            data_folder = dataset_folder + "/local_frame/";
            name_prefix = "Hokuyo_";
        }
        else if (frame_type == FrameType::GLOBAL)
        {
            data_folder = dataset_folder + "/global_frame/";
            name_prefix = "PointCloud";
        }

        path ethz(data_folder.c_str());
        int count = 0;
        for (auto &p : directory_iterator(ethz))
        {
            if (is_regular_file(p.path()) && p.path().filename().string().find(name_prefix) == 0)
            {
                string curr_file = p.path().filename().string();
                size_t last_ind = curr_file.find_last_of(".");
                string raw_name = name_prefix + std::to_string(count);
                count++;
                names.push_back(raw_name);
            }
        }
        sort(names.begin(), names.end());
        cout << "Ethz contains " << names.size() << " files\n";
    }

    EthzHandler::~EthzHandler() {}

    int EthzHandler::read_next_lidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pc)
    {
        string file_path = data_folder + "/" + names[curr_index] + ".csv";
        std::ifstream infile(file_path);
        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return -1;
        }

        std::string line;
        std::getline(infile, line); // skip header line
        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ','))
            {
                tokens.push_back(token);
            }

            pcl::PointXYZI point;
            point.x = std::stof(tokens[1]);
            point.y = std::stof(tokens[2]);
            point.z = std::stof(tokens[3]);
            point.intensity = std::stof(tokens[4]);
            pc->push_back(point);
        }

        infile.close();
        return 0;
    }

    void EthzHandler::read_ground_truth_poses(std::vector<double> &timestamp, std::vector<Eigen::Matrix4d> &poses)
    {
        string file_path = dataset_folder + "/global_frame/pose_scanner_leica.csv";
        std::ifstream infile(file_path);
        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return;
        }
        std::string line;
        std::getline(infile, line); // skip header line
        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ','))
            {
                tokens.push_back(token);
            }
            double curr_timestamp = std::stod(tokens[1]);
            Eigen::Matrix4d curr_pose;
            curr_pose << std::stod(tokens[2]), std::stod(tokens[3]), std::stod(tokens[4]), std::stod(tokens[5]),
                std::stod(tokens[6]), std::stod(tokens[7]), std::stod(tokens[8]), std::stod(tokens[9]),
                std::stod(tokens[10]), std::stod(tokens[11]), std::stod(tokens[12]), std::stod(tokens[13]),
                std::stod(tokens[14]), std::stod(tokens[15]), std::stod(tokens[16]), std::stod(tokens[17]);
            timestamp.push_back(curr_timestamp);
            poses.push_back(curr_pose);
        }
    }

    void EthzHandler::next_frame_index()
    {
        curr_index++;
    }

    void EthzHandler::set_start_index(int start)
    {
        curr_index = start;
    }

    int EthzHandler::get_current_index()
    {
        return curr_index;
    }

    int EthzHandler::get_total_number()
    {
        return names.size();
    }
} // namespace cvo