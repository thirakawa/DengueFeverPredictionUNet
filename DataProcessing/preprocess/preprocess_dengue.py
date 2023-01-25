#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from datetime import date

from check_missing_file import search_missing_files
from config import *


def load_dengue_source():
    input_filename = os.path.join(ORIGINAL_DATA_SOURCE_DIR,
                                  "dengue", "Dengue_Daily.csv")

    # load csv file
    source_data = pd.read_csv(input_filename)

    # extract specific data
    dengue_date = source_data['發病日'].values.tolist()
    x_coords = source_data['最小統計區中心點X'].values.tolist()
    y_corrds = source_data['最小統計區中心點Y'].values.tolist()

    # remove None value data
    dst_date, dst_x, dst_y = [], [], []

    for d, x, y in zip(dengue_date, x_coords, y_corrds):
        if x == 'None' or y == 'None':
            continue
        dst_date.append(d.replace('/', ''))
        dst_x.append(float(x))
        dst_y.append(float(y))

    return dst_date, dst_x, dst_y


def preprocess_dengue():

    # set directory to store processed data
    dengue_dst_dir = os.path.join(DATASET_DIR, "dengue")
    dengue_bin_dst_dir = os.path.join(DATASET_DIR, "dengue_binary")

    # make dirs
    os.makedirs(dengue_dst_dir, exist_ok=True)
    os.makedirs(dengue_bin_dst_dir, exist_ok=True)

    # read csv file
    dd, xx, yy = load_dengue_source()

    # collect unique date values
    date_set = set(dd)

    # process data for each date
    for today in date_set:
        print("process dengue data at %s ..." % today)
        # extract today's data
        today_x, today_y = [], []
        for d, x, y in zip(dd, xx, yy):
            if d == today:
                today_x.append(x)
                today_y.append(y)

        # 1. numerical label data ---------------
        # make 2d map by utilizing np.histogram2d
        # NOTE: x is vertical (latitude, ido) and y is horizontal (longitude, keido)
        dengue_map, _, _ = np.histogram2d(x=today_y, y=today_x, bins=(MAP_SIZE[1], MAP_SIZE[0]),
                                          range=[LAT_RANGE, LON_RANGE])
        # NOTE : flip vertically (for inverting latitude, ido)
        dengue_map = np.flipud(dengue_map)
        dengue_map = dengue_map.astype(np.uint32)
        # save data
        save_name = os.path.join(dengue_dst_dir, today + ".npy")
        np.save(save_name, dengue_map)

        # 2. binary label data ------------------
        # make 2d binary map
        dengue_bin_map = np.zeros((MAP_SIZE[1], MAP_SIZE[0]), dtype=np.bool)
        dengue_bin_map[dengue_map.astype(np.int) != 0] = True
        # save data
        bin_save_name = os.path.join(dengue_bin_dst_dir, today + ".npy")
        np.save(bin_save_name, dengue_bin_map)

    print("process dengue data; done.\n\n")


    # check missing dengue data ###########################
    print("Next, start to interpolate missing dengue data.")

    missing_numerical = search_missing_files(dengue_dst_dir,
                                             date(YEAR_RANGE[0], 1, 1), date(YEAR_RANGE[1], 12, 31))
    missing_binary = search_missing_files(dengue_bin_dst_dir,
                                          date(YEAR_RANGE[0], 1, 1), date(YEAR_RANGE[1], 12, 31))
    # interpolate missing numerical label files
    missing_map_num = np.zeros((MAP_SIZE[1], MAP_SIZE[0]), dtype=np.uint32)
    for miss_num in missing_numerical:
        miss_save_name_num = os.path.join(dengue_dst_dir, miss_num + ".npy")
        np.save(miss_save_name_num, missing_map_num)
    # interpolate missing binary label files
    missing_map_bin = np.zeros((MAP_SIZE[1], MAP_SIZE[0]), dtype=np.bool)
    for miss_bin in missing_binary:
        miss_save_name_bin = os.path.join(dengue_bin_dst_dir, miss_bin + ".npy")
        np.save(miss_save_name_bin, missing_map_bin)

    print("Interpolate missing data; done.\n\n")


if __name__ == "__main__":
    preprocess_dengue()
