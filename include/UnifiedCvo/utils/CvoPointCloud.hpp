#pragma once
#include <string>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "utils/data_type.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <utils/CvoPoint.hpp>


namespace semantic_bki {
  class SemanticBKIOctoMap;
}

namespace pcl {
  struct CvoPoint;
}

extern template struct pcl::PointSegmentedDistribution<FEATURE_DIMENSIONS, NUM_CLASSES>;

namespace cvo {
  
  template <typename DepthType> class ImageRGBD;
  class ImageStereo;
  class RawImage;
  class Calibration;

  class
  //#ifdef __CUDACC__
  // __align__(16)
  //#else
  //  alignas(16)
  // #endif    
 CvoPointCloud{
  public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    enum PointSelectionMethod {
      CV_FAST,
      RANDOM,
      DSO_EDGES,
      DSO_EDGES_WITH_RANDOM,
      LIDAR_EDGES,
      CANNY_EDGES,
      EDGES_ONLY,
      LOAM,
      FULL
    };

    enum GeometryType {
      EDGE,
      SURFACE
    };

    const int pixel_pattern[8][2] = {{0,0}, {-1, 0},{-1,-1}, {-1,1}, {0,1},{0,-1},{1,1},{1,0} };

    /// Constructor for stereo image
    CvoPointCloud(const ImageStereo & left_raw_image,
                  const Calibration &calib,
                  PointSelectionMethod pt_selection_method=CV_FAST);
    

    /// Constructor for rgbd image
    template <typename DepthType>
    CvoPointCloud(const ImageRGBD<DepthType> & rgb_raw_image,
                  const Calibration &calib,
                  PointSelectionMethod pt_selection_method=CV_FAST);

    /// Constructor for lidar input
    CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                  int target_num_points,
                  int beam_num,
                  PointSelectionMethod pt_selection_method=LOAM);

    
    /// construtor for lidar points with semantics
    CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, 
                  const std::vector<int> & semantics,
                  int num_classes,
                  int target_num_points,
                  int beam_num,
                  PointSelectionMethod pt_selection_method=LOAM);

    CvoPointCloud(int feature_dimensions, int num_classes);

    //CvoPointCloud(const std::string & filename);

    CvoPointCloud();
    ~CvoPointCloud();    


    // overloaded operators and copy constructors
    CvoPointCloud(const CvoPointCloud & to_copy);
    CvoPointCloud & operator+=(const CvoPointCloud & b);
    friend CvoPointCloud operator+(CvoPointCloud a, const CvoPointCloud & b);
    CvoPointCloud & operator=(const CvoPointCloud& input);

    
    template <typename PointT>
      CvoPointCloud(const pcl::PointCloud<PointT> & pc);    
    template <typename PointT>
      CvoPointCloud(const pcl::PointCloud<PointT> & pc, GeometryType g_type);    

    /*
    CvoPointCloud(pcl::PointCloud<pcl::PointXYZIR>::Ptr pc,
                  int target_num_points = 5000
                  );



    CvoPointCloud(pcl::PointCloud<pcl::PointXYZIR>::Ptr pc, 
                  const std::vector<int> & semantics,
                  int num_classes=19,
                  int target_num_points = 5000
                  );
    */
    // Constructor from continuous maps
    CvoPointCloud(const semantic_bki::SemanticBKIOctoMap * map,
                  int num_semantic_class);


    int read_cvo_pointcloud_from_file(const std::string & filename);

    static void transform(const Eigen::Matrix4f& pose,
                          const CvoPointCloud & input,
                          CvoPointCloud & output);

    // setter
    void erase(size_t index);

    // getters
    std::vector<cvo::CvoPoint> get_points() const {return points_;}
    const cvo::CvoPoint & point_at(unsigned int index) const {return points_.at(index);}
    int num_points() const {return num_points_;}
    int size() const {return num_points_;}
    int num_classes() const {return num_classes_;}
    int num_features() const {return feature_dimensions_;}
    int feature_dimensions() const {return feature_dimensions_;}
    int num_geometric_types() const {return num_geometric_types_; }
    const std::vector<Eigen::Vector3f> positions() const {
      std::vector<Eigen::Vector3f> positions;
      for(int j = 0; j < num_points_; j++){
        positions.push_back(Eigen::Vector3f(points_[j].x, points_[j].y, points_[j].z));
      }
      return positions;
    }
    Eigen::Vector3f at(unsigned int index) const;
    Eigen::Vector3f xyz_at(unsigned int index) const;
    const Eigen::VectorXf label_at(unsigned int index) const { return Eigen::Map<const Eigen::VectorXf>(points_[index].label_distribution, NUM_CLASSES); }
    const Eigen::VectorXf feature_at(unsigned int index) const { return Eigen::Map<const Eigen::VectorXf>(points_[index].features, FEATURE_DIMENSIONS); }
    Eigen::Vector2f geometry_type_at(unsigned int index) const;
    
    //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals() const {return normals_;}
    //const Eigen::Matrix<float, Eigen::Dynamic, 9> & covariance() const {return covariance_;}
    //const pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals() const {return cloud_with_normals_;}
    //const Eigen::Matrix<float, Eigen::Dynamic, 2> & types() const {return types_;}
    //const std::vector<float> & covariance()  const {return covariance_;}
    //const std::vector<float> & eigenvalues() const {return eigenvalues_;}

    // IO helpers
    template<typename PointT> void export_to_pcd(pcl::PointCloud<PointT> & output) const;
    void write_to_color_pcd(const std::string & name) const;
    void write_to_label_pcd(const std::string & name) const;
    void write_to_pcd(const std::string & name) const;
    void write_to_txt(const std::string & name) const;
    void write_to_intensity_pcd(const std::string & name) const;
    
    void reserve(int num_points, int feature_dims, int num_classes);
    int add_point(int index, const Eigen::Vector3f & xyz, const Eigen::VectorXf & feature, const Eigen::VectorXf & label, const Eigen::VectorXf & geometric_type);
    void push_back(const cvo::CvoPoint & new_point);
   
  private:
    int num_points_;
    int num_classes_;
    int feature_dimensions_;
    int num_geometric_types_;

    std::vector<cvo::CvoPoint> points_;
    //std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> positions_;  // points position. x,y,z
//    std::vector<Eigen::Vector3f> positions_;
    //Eigen::Matrix<float, Eigen::Dynamic, 3> positions_;
//    Eigen::MatrixXf features_;
    //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> features_;   // rgb, gradient in [0,1]
    //std::vector<Eigen::Matrix<float, 1, Eigen::Dynamic>> features_;   // rgb, gradient in [0,1]
    //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> normals_;  // surface normals
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels_; // number of points by number of classes
    //std::vector<Eigen::Matrix<float, 1, Eigen::Dynamic>> labels_;   // rgb, gradient in [0,1]    
//    std::vector<float> geometric_types_;
    
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals_;
    //Eigen::Matrix<float, Eigen::Dynamic, 2> types_; // type of the point using loam point selector, edge=(1,0), surface=(0,1)
    
    cv::Vec3f avg_pixel_color_pattern(const cv::Mat & raw, int u, int v, int w);
    bool check_type_consistency() const;    


    
    //std::vector<float> covariance_;
    //std::vector<float> eigenvalues_;
    //thrust::device_vector<float> eigenvalues_;
    //perl_registration::cuPointCloud<CvoPoint>::SharedPtr pc_gpu;
    //void compute_covarianes(pcl::PointCloud<pcl::PointXYZI> & pc_raw);
    //void compute_covariance(const pcl::PointCloud<pcl::PointXYZI> & pc_input,
    //                        // outputs
    //                        std::vector<float>& covariance_all,
    //                        std::vector<float>& eigenvalues_all) const;


  };
  // for historical reasons
  typedef CvoPointCloud point_cloud;



  void write_all_to_label_pcd(const std::string name,
                          const pcl::PointCloud<pcl::PointXYZI> & pc,
                          int num_class,
                          const std::vector<int> & semantic);

}
