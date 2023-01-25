#!/bin/bash


python3 make_demo_video.py\
    -d /raid/hirakawa/Dengue/DenguePrediction/unet_segmentation/runs/all/demo_result \
    -o all_test.mp4

python3 make_demo_video.py\
    -d /raid/hirakawa/Dengue/DenguePrediction/unet_segmentation/runs/precipitation/demo_result \
    -o precipitation_test.mp4

python3 make_demo_video.py\
    -d /raid/hirakawa/Dengue/DenguePrediction/unet_segmentation/runs/solar/demo_result \
    -o solar_test.mp4

python3 make_demo_video.py\
    -d /raid/hirakawa/Dengue/DenguePrediction/unet_segmentation/runs/sst_day/demo_result \
    -o sst_day_test.mp4

python3 make_demo_video.py\
    -d /raid/hirakawa/Dengue/DenguePrediction/unet_segmentation/runs/precipitation_solar/demo_result \
    -o precipitation_solar_test.mp4
