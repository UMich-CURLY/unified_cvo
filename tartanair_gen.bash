#/bin/bash

cd build 
make -j 
cd ..
firstframe=190
for startframe in ${firstframe} 210 230 250
do
    ./build/bin/cvo_export_pointcloud_tartan ./P001  cvo_params/cvo_intensity_params_irls_exp_tartanair.yaml ${startframe} ~/demo_data/ ${firstframe}
done
