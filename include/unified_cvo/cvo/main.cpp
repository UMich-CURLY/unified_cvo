#include "rkhs_se3.hpp"

void load_file_name(string assoc_pth, vector<string> &vstrRGBName, \
                    vector<string> &vstrRGBPth, vector<string> &vstrDepPth);

int main(int argc, char** argv){
    
    // string dataset = "freiburg3_structure_notexture_far";
    string dataset = "freiburg1_desk";
    int dataset_seq = 1;

    // downsampled pcd from tum rgbd dataset
    string folder = "../../../../data/rgbd_dataset/" + dataset + "/";
    // string folder = "/media/justin/LaCie/data/rgbd_dataset/" + dataset + "/";
    string ds_folder = folder + "pcd_ds/";
    string dso_folder = folder + "pcd_dso/";
    string dense_folder = folder + "pcd_full/";
    string assoc_pth = folder + "assoc.txt";
    

    // create our registration class
    cvo::cvo cvo;

    // load associate file
    vector<string> vstrRGBName;     // vector for image names
    vector<string> vstrRGBPth;
    vector<string> vstrDepPth;
    load_file_name(assoc_pth,vstrRGBName,vstrRGBPth,vstrDepPth);
    int num_img = vstrRGBName.size();
    // std::cout<<"num images: "<<num_img<<std::endl;
    // num_img = 2;

    // export as transform matrix
    // ofstream fPoseCsv;
    // fPoseCsv.open(folder+"cvo_poses.csv");
    // fPoseCsv << "frame1, frame2, tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33" << endl;

    // export as quarternion
    ofstream fPoseQtTxt;
    fPoseQtTxt.open(folder+"cvo_poses_qt.txt");
    // fPoseQtTxt << "index, file_name, qw, qx, qy, qz, tx, ty, tz" << endl;
    boost::timer::cpu_timer total_time;
    // loop through all the images in associate files
    for(int i=0;i<num_img;i++){
        string pcd_pth = ds_folder + vstrRGBName[i] + ".pcd";
        string pcd_dso_pth = dso_folder + vstrRGBName[i] + ".pcd";

        string RGB_pth = folder + vstrRGBPth[i];
        string dep_pth = folder + vstrDepPth[i];
        
        if(cvo.init){
            std::cout<<"----------------------"<<std::endl;
            std::cout<<"Processing frame "<<i<<std::endl;
            std::cout<<"Aligning " + vstrRGBName[i-1] + " and " + vstrRGBName[i] <<std::endl;
        }
        else{
            std::cout<<"initializing..."<<std::endl;
        }

        boost::timer::cpu_timer timer;
        
        if(cvo.init == false){
            cvo.set_pcd(dataset_seq, pcd_pth, RGB_pth, dep_pth, pcd_dso_pth);
            continue;
        }
        

        cvo.set_pcd(dataset_seq, pcd_pth, RGB_pth, dep_pth, pcd_dso_pth);
        cvo.align();
            
        
            
        
        std::cout<<"elapse time: "<<timer.format()<<std::endl;
        std::cout<<"Total iterations: "<<cvo.iter<<std::endl;
        std::cout<<"RKHS-SE(3) Object Transformation Estimate: \n"<<cvo.transform.matrix()<<std::endl;
        
        // log out transformation matrix for each frame
        // fPoseCsv << i << "," << i+1 << "," \
        //         << cvo.transform(0,3) << "," << cvo.transform(1,3) << "," << cvo.transform(2,3) << ","\
        //         << cvo.transform(0,0) << "," << cvo.transform(0,1) << "," << cvo.transform(0,2) << ","\
        //         << cvo.transform(1,0) << "," << cvo.transform(1,1) << "," << cvo.transform(1,2) << ","\
        //         << cvo.transform(2,0) << "," << cvo.transform(2,1) << "," << cvo.transform(2,2) << std::endl;

        // log out quaternion
        Eigen::Quaternionf q(cvo.accum_transform.matrix().block<3,3>(0,0));
        fPoseQtTxt<<vstrRGBName[i]<<" ";
        fPoseQtTxt<<cvo.accum_transform(0,3)<<" "<<cvo.accum_transform(1,3)<<" "<<cvo.accum_transform(2,3)<<" "; 
        fPoseQtTxt<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
        
    }


    std::cout<<"======================="<<std::endl;
    std::cout<<"Total time for "<<num_img<<" frames is: "<<total_time.format()<<std::endl;
    std::cout<<"======================="<<std::endl;

    // fPoseCsv.close();
    fPoseQtTxt.close();
}

void load_file_name(string assoc_pth, vector<string> &vstrRGBName, \
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
