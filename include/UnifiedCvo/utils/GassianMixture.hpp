#pragma once
#include <random>
#include <Eigen/Dense>
namespace cvo {
  
  // pdf:  f(x) = gamma * N(p, sigma) + (1- gamma) * uniform(start, end)  
  class GaussianMixtureDepthGenerator {
  public:
    GaussianMixtureDepthGenerator(float gamma,
                                  float sigma,
                                  float uniform_range): rd_(), gen_(rd_()),
                                                bernoulli_dist_(gamma),
                                                normal_dist_{0, sigma},
                                                        uniform_{-uniform_range, uniform_range}{}

    ~GaussianMixtureDepthGenerator() {}

    Eigen::Vector3f sample(const Eigen::Vector3f & center,
                           const Eigen::Vector3f & dir,
                           bool * is_inlier = nullptr) {
      float g = 0;
      if (bernoulli_dist_(gen_)) {
        /// gaussian
        g = normal_dist_(gen_);
        if (is_inlier ) *is_inlier = true;
      } else {
        g = uniform_(gen_);
        if (is_inlier) *is_inlier = false;
      }
      Eigen::Vector3f sampled = center + dir * g;
      return sampled;
    }

  private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::bernoulli_distribution bernoulli_dist_;
    std::uniform_real_distribution<> uniform_;
    std::normal_distribution<> normal_dist_;
  };
}
