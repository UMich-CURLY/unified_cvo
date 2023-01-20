#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"

using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
    // list all files in current directory.
    //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
//  cvo::KittiHandler kitti(argv[1], 0);
    cvo::TartanAirHandler tartan(argv[1]);
    int total_iters = tartan.get_total_number();
    string cvo_param_file(argv[2]);
    string calib_file;
    calib_file = string(argv[1] ) +"/cvo_calib.txt";
    cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);
    std::ofstream accum_output(argv[3]);
    int start_frame = std::stoi(argv[4]);
//  kitti.set_start_index(start_frame);
    tartan.set_start_index(start_frame);
    int max_num = std::stoi(argv[5]);

    accum_output <<"1 0 0 0 0 1 0 0 0 0 1 0\n";

    cvo::CvoGPU cvo_align(cvo_param_file );
    cvo::CvoParams & init_param = cvo_align.get_params();
    float ell_init = init_param.ell_init;
    float ell_decay_rate = init_param.ell_decay_rate;
    int ell_decay_start = init_param.ell_decay_start;
    init_param.ell_init = init_param.ell_init_first_frame;
    init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
    init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
    cvo_align.write_params(&init_param);

    std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

    //cvo::cvo cvo_align_cpu("/home/rayzhang/outdoor_cvo/cvo_params/cvo_params.txt");

    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
    //init_guess(2,3)=2.22;
    Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();

    cv::Mat source_left, source_right;
    //std::vector<float> semantics_source;
    //kitti.read_next_stereo(source_left, source_right, 19, semantics_source);
//  kitti.read_next_stereo(source_left, source_right);
    tartan.read_next_stereo(source_left, source_right);
    std::cout<<"read source raw...\n";
    std::shared_ptr<cvo::ImageStereo> source_raw(new cvo::ImageStereo(source_left, source_right));
    std::cout<<"build source CvoPointCloud...\n";
    std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_raw, calib
            //, cvo::CvoPointCloud::CANNY_EDGES
    ));

    std::cout<<"write to opcd\n";
    source->write_to_color_pcd("source.pcd");
    //source->write_to_color_pcd(std::to_string(start_frame) + ".pcd");

    double total_time = 0;
    int i = start_frame;
    for (; i<min(total_iters, start_frame+max_num)-1 ; i++) {

        // calculate initial guess
        std::cout<<"\n\n\n\n============================================="<<std::endl;
        std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;

//    kitti.next_frame_index();
        tartan.next_frame_index();
        cv::Mat left, right;
        //vector<float> semantics_target;
        //if (kitti.read_next_stereo(left, right, 19, semantics_target) != 0) {
        if (tartan.read_next_stereo(left, right) != 0) {
            std::cout<<"finish all files\n";
            break;
        }

        std::shared_ptr<cvo::ImageStereo> target_raw(new cvo::ImageStereo(left, right));
        std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_raw, calib
                //,cvo::CvoPointCloud::CANNY_EDGES
        ));

        target->write_to_color_pcd("target.pcd");
        //target->write_to_color_pcd(std::to_string(i+1) + ".pcd");
        Eigen::Matrix4f result, init_guess_inv;
        Eigen::Matrix4f identity_init = Eigen::Matrix4f::Identity();
        init_guess_inv = init_guess.inverse();

        //double in_product_pre_t2s = cvo_align.function_angle(*source, *target, init_guess_inv, ell_init, false);
        //std::cout<<"The init guess function_angle from frame "<<i <<" to "<< i-1 <<" is "<<in_product_pre_t2s<<"\n";

        //double in_product_identity = cvo_align.function_angle(*source, *target, identity_init, ell_init);
        //std::cout<<"The identity guess  inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_identity<<"\n";


        printf("Start align... num_fixed is %d, num_moving is %d\n", source->num_points(), target->num_points());
        std::cout<<std::flush;

        double this_time = 0;
        cvo_align.align(*source, *target, init_guess_inv, result, nullptr,&this_time);
        total_time += this_time;
        //cvo_align.align(*source, *target, init_guess, result);

        // get tf and inner product from cvo getter
        //double in_product = cvo_align.inner_product_cpu(*source, *target, result, ell_init);
        //double in_product = cvo_align.function_angle(*source, *target, result.inverse(), ell_init, false);

        //double in_product_normalized = cvo_align.inner_product_normalized();a
        //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
        //std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
        //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
        std::cout<<"Transform is "<<result <<"\n\n";

        // append accum_tf_list for future initialization
        init_guess = result;
        accum_mat = accum_mat * result;
        std::cout<<"accum tf: \n"<<accum_mat<<std::endl;

        if (i==start_frame) {
            cvo::CvoPointCloud t_target;
            cvo::CvoPointCloud::transform(result, *target, t_target);
            t_target.write_to_color_pcd("t_target.pcd");
            /*
            cvo::Association association;
            Eigen::Matrix4f T_t2s = init_guess.inverse();
            std::cout<<"compute association\n";
            Eigen::Matrix3f kernel;
            kernel << 0.5 , 0, 0,
              0, 0.5, 0,
              0,   0, 0.6;
             cvo_align.compute_association_gpu(*source, *target, T_t2s, kernel, association );
            */
        }

        // log accumulated pose

        accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                     <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                     <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
        accum_output<<"\n";
        accum_output<<std::flush;

        std::cout<<"\n\n===========next frame=============\n\n";

        source = target;
        if (i == start_frame) {
            init_param.ell_init = ell_init;
            init_param.ell_decay_rate = ell_decay_rate;
            init_param.ell_decay_start = ell_decay_start;
            cvo_align.write_params(&init_param);
        }

    }

    std::cout<<"Average registration time is "<<total_time / (i - start_frame + 1)<<std::endl;

    accum_output.close();

    return 0;
}
