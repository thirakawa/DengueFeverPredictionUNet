#!/bin/bash


python3 train.py --data_dir /raid/hirakawa/Dengue/DengueDataset/2weeks \
    --logdir ./runs/all \
    --delay 0 \
    --sst_type day --use_precipitation --use_solar \
    --gpu_id 0


python3 demo.py --data_dir /raid/hirakawa/Dengue/DengueDataset/2weeks \
    --logdir ./runs/5_all \
    --gpu_id 0
