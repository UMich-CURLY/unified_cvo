cd build
make -j
cd ..
./build/bin/cvo_irls_rand_pcd demo_data/bunny.pcd cvo_params/cvo_intensity_params_irls_bunny.yaml 4 0.4 1 1 0.2 0.02 0.5
