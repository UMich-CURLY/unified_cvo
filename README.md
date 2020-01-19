# Outdoor CVO

### Dockerfile to help resolve dependencies
[Docker file for building CVO](https://github.com/UMich-BipedLab/docker_images/tree/master/cvo_gpu)


### Dependencies
*  `cuda10.2`
*  `gcc7`
*  `SuiteParse`
* `Sophus 1.0.0`
* `Eigen 3.3.7`
* `TBB`
* `Boost`
* `pcl1.9.1`
* `OpenCV3` or `OpenCV4`
* `GTSAM` (`master branch`)

### Compile
```
mkdir build
cd build
cmake ..
make -j
```
