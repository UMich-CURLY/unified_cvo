#!/bin/bash

rm -f log.txt
 ./build/bin/cvo_kitti_stereo_semantic /run/media/rayzhang/Samsung_T5/kitti/05_raw_imgs/05/ /run/media/rayzhang/Samsung_T5/kitti/05_raw_imgs/05/cvo_calib.txt 19 0 2>&1 | tee log.txt

