#include <algorithm>
#include <chrono>
#include <pcl/filters/voxel_grid.h>

#include "mapping/bkioctomap.h"
#include "mapping/bki.h"
#include "utils/CvoPointCloud.hpp"
using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace cvo {
    
  CvoPointCloud::CvoPointCloud(const semantic_bki::SemanticBKIOctoMap * map,
                               const int num_classes) {
    num_classes_ = num_classes;
    int num_point_counter = 0;
    std::vector<std::vector<float> > features;
    std::vector<std::vector<float> > labels;
    positions_.reserve(65536);
    features.reserve(65536);
    labels.reserve(65536);
    feature_dimensions_ = 5;
    
    for (auto it = map->begin_leaf(); it != map->end_leaf(); ++it) {
      if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
        // position
        semantic_bki::point3f  p = it.get_loc();
        Vec3f xyz;
        xyz << p.x(), p.y(), p.z();
        positions_.push_back(xyz);
               
        // features
        if(feature_dimensions_==5){
          std::vector<float> feature(5, 0);
          it.get_node().get_features(feature);
          features.push_back(feature);
        }
        else if(feature_dimensions_==1){
          std::vector<float> feature_1(1, 0);
          it.get_node().get_features(feature_1);
          features.push_back(feature_1);
        }
        
        // labels
        std::vector<float> label(num_classes_, 0);
        it.get_node().get_occupied_probs(label);
        labels.push_back(label);
        num_point_counter++;
      }
    }
      
    num_points_ = num_point_counter ;
    features_.resize(num_points_, feature_dimensions_);
    labels_.resize(num_points_, num_classes);

    for (int i = 0; i < num_points_; i++) {
      //memcpy(labels_.data()+ num_classes * sizeof(float) * i, labels[i].data(), num_classes * sizeof(float));
      labels_.row(i) = Eigen::Map<VecXf_row>(labels[i].data(), num_classes);
      if(feature_dimensions_==5){
        features_.row(i) = Eigen::Map<Vec5f_row>(features[i].data());
      }
      else if(feature_dimensions_==1){
        features_(i,0) = *features[i].data();
      }

    }
    //std::cout<<"Read labels from map:\nlabel" << labels_.row(0)<<"\n"<<labels_.row(num_points_-1)<<", color: ";
    //std::cout<< features_.row(0)<<"\n"<<features_.row(num_points_-1)<<"\n";
  }

  
}

namespace semantic_bki {

    SemanticBKIOctoMap::SemanticBKIOctoMap() : SemanticBKIOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        3,  // num_class
                                        1.0, // sf2
                                        1.0, // ell
                                        1.0f, // prior
                                        1.0f, // var_thresh
                                        0.3f, // free_thresh
                                        0.7f // occupied_thresh
                                    ) { }

    SemanticBKIOctoMap::SemanticBKIOctoMap(float resolution,
                        unsigned short block_depth,
                        int num_class,
                        float sf2,
                        float ell,
                        float prior,
                        float var_thresh,
                        float free_thresh,
                        float occupied_thresh)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);
        
        // Bug fixed
        Block::cell_num = static_cast<unsigned short>(round(Block::size / Block::resolution));
        std::cout << "block::resolution: " << Block::resolution << std::endl;
        std::cout << "block::size: " << Block::size << std::endl;
        std::cout << "block::cell_num: " << Block::cell_num << std::endl;
        
        SemanticOcTree::max_depth = block_depth;

        SemanticOcTreeNode::num_class = num_class;
        SemanticOcTreeNode::sf2 = sf2;
        SemanticOcTreeNode::ell = ell;
        SemanticOcTreeNode::prior = prior;
        SemanticOcTreeNode::var_thresh = var_thresh;
        SemanticOcTreeNode::free_thresh = free_thresh;
        SemanticOcTreeNode::occupied_thresh = occupied_thresh;
    }

    SemanticBKIOctoMap::~SemanticBKIOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void SemanticBKIOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        SemanticOcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::insert_pointcloud_csm(const CVOPointCloud *cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {

      //#ifdef DEBUG
        auto start = std::chrono::system_clock::now();
        std::cout<<"Insert pointcloud: " << "cloud size: " << cloud->num_points() << " origin: " << origin<<"\n";
        //#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);
        //#ifdef DEBUG
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        //Debug_Msg("Got training data, size: " << xy.size()<<", takes "<<elapsed_seconds.count()<<" seconds" );
        std::cout<<"Got training data, size: " << xy.size()<<", takes "<<elapsed_seconds.count()<<" seconds\n" ;
        start = std::chrono::system_clock::now();
        //#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        start = std::chrono::system_clock::now();
        //Debug_Msg("Training done, takes "<<elapsed_seconds.count()<<" seconds");
        std::cout<<"after get_blocks_in_bbox, takes "<<elapsed_seconds.count()<<" seconds\n";

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        start = std::chrono::system_clock::now();
        //Debug_Msg("Training done, takes "<<elapsed_seconds.count()<<" seconds");
        std::cout<<"after insert, takes "<<elapsed_seconds.count()<<" seconds\n";
	
	#ifdef OPENMP
	std::cout<<"openmp enalbed\n";
#endif
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;        
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y, block_f;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second[0]);  // label
                block_f.push_back(it->second[1]);  // r
                block_f.push_back(it->second[2]);  // g
                block_f.push_back(it->second[3]);  // b
                block_f.push_back(it->second[4]);  // dx
                block_f.push_back(it->second[5]);  // dy
            }
          

            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y, block_f);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }
        //#ifdef DEBUG
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        //Debug_Msg("Training done, takes "<<elapsed_seconds.count()<<" seconds");
        std::cout<<"Training done, takes "<<elapsed_seconds.count()<<" seconds\n";
        start = std::chrono::system_clock::now();
        std::cout<<"Prediction: block number: " << test_blocks.size()<<"\n";
        //#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

	          // For counting sensor model
            auto bgk = bgk_arr.find(key);
            if (bgk == bgk_arr.end())
              continue;

            vector<vector<float>> ybars;
            vector<vector<float>> fbars;
            bgk->second->predict_csm(xs, ybars, fbars);

            int j = 0;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                SemanticOcTreeNode &node = leaf_it.get_node();

                // Only need to update if kernel density total kernel density est > 0
                node.update(ybars[j], fbars[j]);
            }

        }
        //#ifdef DEBUG
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        //Debug_Msg("Prediction done, predition takes "<<elapsed_seconds.count()<<" seconds");
        std::cout<<"Prediction done, predition takes "<<elapsed_seconds.count()<<" seconds\n";
        //#endif

        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void SemanticBKIOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPPointCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(it->second->get_center(), 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    void SemanticBKIOctoMap::get_training_data(const CVOPointCloud *cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {
        xy.clear();
        for (int i = 0; i < cloud->num_points(); ++i) {
            point3f p(cloud->positions()[i][0], cloud->positions()[i][1], cloud->positions()[i][2]);
            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            
            std::vector<float> properties(6, 0);
            int pix_label;
            cloud->labels().row(i).maxCoeff(&pix_label);
            properties[0] = pix_label + 1;
            for (int j = 0; j < 5; ++j)
              properties[j + 1] = cloud->features()(i, j);
            xy.emplace_back(p, properties);

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
              std::vector<float> properties(6, 0);
              xy.emplace_back(*p, properties);
            }
        }

        point3f p(origin.x(), origin.y(), origin.z());
        std::vector<float> properties(6, 0);
        xy.emplace_back(p, properties);
    }


    void SemanticBKIOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution) const {
        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = free_resolution;
        while (d < l) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d += free_resolution;
        }
        if (l > free_resolution)
            frees.emplace_back(x0 + nx * (l - free_resolution), y0 + ny * (l - free_resolution), z0 + nz * (l - free_resolution));
    }


    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void SemanticBKIOctoMap::bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x());
            y.push_back(it->first.y());
            z.push_back(it->first.z());
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void SemanticBKIOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         GPPointCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         GPPointCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::search_callback, static_cast<void *>(&out));
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::count_callback, NULL);
    }

    bool SemanticBKIOctoMap::count_callback(GPPointType *p, void *arg) {
        return false;
    }

    bool SemanticBKIOctoMap::search_callback(GPPointType *p, void *arg) {
        GPPointCloud *out = static_cast<GPPointCloud *>(arg);
        out->push_back(*p);
        return true;
    }


    int SemanticBKIOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         GPPointCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *SemanticBKIOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
          return SemanticOcTreeNode();
        } else {
          return SemanticOcTreeNode(block->search(p));
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
