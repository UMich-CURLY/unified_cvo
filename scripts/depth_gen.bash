cd build
make -j
cd ..
mkdir -p /home/`whoami`/media/Samsung_T5/finnforest/RectifiedData/S01_8Hz/depth
mkdir -p /home/`whoami`/media/Samsung_T5/finnforest/RectifiedData/S01_8Hz/pcd
./build/bin/depth_gen finnforest /home/`whoami`/media/Samsung_T5/finnforest/RectifiedData/S01_8Hz/ /home/`whoami`/media/Samsung_T5/forest/RectifiedData/S01_8Hz/depth /home/`whoami`/media/Samsung_T5/finnforest/RectifiedData/S01_8Hz/pcd 0 2
