#include <iostream>
#include <memory>
#include <random>
#include <time.h>
//#include "gtest/gtest.h"

#include <pcl/kdtree/kdtree_flann.h>
#include <thrust/device_vector.h>
#include "pcl/point_cloud.h"

#include "cukdtree/cukdtree.h"
#include "cupointcloud/cupointcloud.h"

// Test kd tree look up
//TEST(cuKdTreeTest, MatchesPCLKdTree) {
void run_test() {
  std::uniform_real_distribution<> d(-30, 30);
  std::random_device rd;
  std::mt19937 gen(rd());
  pcl::PointCloud<pcl::PointXYZ> pc;

  for (size_t i = 0; i < 12500; i++) {
    pcl::PointXYZ p(d(gen), d(gen), d(gen));
    pc.push_back(p);
  }
  int loops = 100;

  auto d_pc = std::make_shared<
      perl_registration::cuPointCloud<perl_registration::cuPointXYZ>>(pc);

  std::cout<<"cukdtree:\n";
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  int r = -1;
  perl_registration::cuKdTree<perl_registration::cuPointXYZ> kd_tree;
  kd_tree.SetInputCloud(d_pc);
  
  for (int i = 0 ; i < loops; i++) {
    
    pcl::PointCloud<pcl::PointXYZ> query;
    pcl::PointXYZ q(d(gen), d(gen), d(gen));
    query.push_back(q);
    auto d_query = std::make_shared<
      perl_registration::cuPointCloud<perl_registration::cuPointXYZ>>(query);
    
    thrust::device_vector<int> results;
    kd_tree.NearestKSearch(d_query, KDTREE_K_SIZE, results);
    r = results[0];
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime2d, totalTime2d;
  cudaEventElapsedTime(&elapsedTime2d, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  totalTime2d = elapsedTime2d/(1000*loops);
  printf("2D: thrust time = %f\n", totalTime2d);

  perl_registration::cuPointXYZ p = d_pc->points[r];

  std::cout << "cuKdTREE\n";
  std::cout << "Nearest Neighbor is Index " << r
            << " and has cordinates x: " << p.x << " y: " << p.y
            << " z: " << p.z << std::endl;
  /*
  std::cout<<"PCL\n";
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud(pc.makeShared());

  // K nearest neighbor search
  int K = KDTREE_K_SIZE;

  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);

  if (kdtree.nearestKSearch(q, K, pointIdxNKNSearch, pointNKNSquaredDistance) >
      0) {
    std::cout << "pcl::KdTREE\n";
    for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i) {
      std::cout << " Nearest Neighbor is Index " << pointIdxNKNSearch[i]
                << " and has cordinates x: "
                << pc.points[pointIdxNKNSearch[i]].x
                << " y: " << pc.points[pointIdxNKNSearch[i]].y
                << " x: " << pc.points[pointIdxNKNSearch[i]].z << std::endl;
    }
  }
  //ASSERT_FLOAT_EQ(p.x, pc.points[pointIdxNKNSearch[0]].x);
  //ASSERT_FLOAT_EQ(p.y, pc.points[pointIdxNKNSearch[0]].y);
  //ASSERT_FLOAT_EQ(p.z, pc.points[pointIdxNKNSearch[0]].z);
  */
}

int main(int argc, char **argv) {
  //::testing::InitGoogleTest(&argc, argv);
  //return RUN_ALL_TESTS();
  run_test();
  return 0;
}
