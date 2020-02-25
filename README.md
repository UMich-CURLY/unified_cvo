# Outdoor CVO

### Dockerfile to help resolve dependencies
[Docker file for building CVO](https://github.com/UMich-BipedLab/docker_images/tree/master/cvo_gpu)

Follow it to first get a cuda10 environment. Then manually install the following dependencies from source inside your docker container.

### Dependencies
*  `cuda10.2` (already in docker)
*  `gcc7` (already in docker)
*  `SuiteParse` (already in docker)
* `Sophus 1.0.0 release` (built from source)
* `Eigen 3.3.7` (built from source)
* `TBB` (already in docker)
* `Boost 1.65` (already in docker)
* `pcl 1.9.1` (built from source)
* `OpenCV3` or `OpenCV4` (already in docker)
* `GTSAM` (`default branch`, built from source)

Note: 'pcl-1.9.1' need to be changed and compiled to get it working with cuda. 
* `pcl/io/boost.h`: add `#include <boost/numeric/conversion/cast.hpp>` at the end of the file before `#endif`
* `pcl/point_cloud.h`: Some meet the error 
```
pcl/point_cloud.h:586100 error: template-id ‘getMapping’ used as a declarator
friend boost::shared_ptr& detail::getMapping(pcl::PointCloud &p);
```
Please see [this dock](https://github.com/autowarefoundation/autoware/issues/2094) for reference

### Compile
```
mkdir build
cd build
cmake ..
make -j
```
