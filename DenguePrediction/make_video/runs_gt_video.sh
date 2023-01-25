#!/bin/bash


python3 make_groundtruth_video.py \
    --data_dir /raid/hirakawa/Dengue/DengueDataset/1day \
    -o /raid/hirakawa/Dengue/DengueDataset/1day/gt_dengue.mp4 --type dengue

python3 make_groundtruth_video.py \
    --data_dir /raid/hirakawa/Dengue/DengueDataset/1day \
    -o /raid/hirakawa/Dengue/DengueDataset/1day/gt_dengue_binary.mp4 --type dengue_binary

python3 make_groundtruth_video.py \
    --data_dir /raid/hirakawa/Dengue/DengueDataset/1day \
    -o /raid/hirakawa/Dengue/DengueDataset/1day/gt_precipitation.mp4 --type precipitation

python3 make_groundtruth_video.py \
    --data_dir /raid/hirakawa/Dengue/DengueDataset/1day \
    -o /raid/hirakawa/Dengue/DengueDataset/1day/gt_solar.mp4 --type solar

python3 make_groundtruth_video.py \
    --data_dir /raid/hirakawa/Dengue/DengueDataset/1day \
    -o /raid/hirakawa/Dengue/DengueDataset/1day/gt_sst-day-interpolated.mp4 --type sst-day-interpolated

python3 make_groundtruth_video.py \
    --data_dir /raid/hirakawa/Dengue/DengueDataset/1day \
    -o /raid/hirakawa/Dengue/DengueDataset/1day/gt_sst-night-interpolated.mp4 --type sst-night-interpolated


