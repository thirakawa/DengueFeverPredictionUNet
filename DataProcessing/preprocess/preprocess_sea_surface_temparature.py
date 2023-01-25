#!/usr/bin/env python3

import os
from glob import glob
import xarray as xa
import numpy as np
import cv2

import multiprocessing as mp

from config import *


def multi_process(source_filenames, sst_save_dir, lon_range, lat_range, map_size, year_thresh, process_num):
    print("start process:", process_num)

    for sfn in source_filenames:
        basename = os.path.basename(sfn)
        date = basename[:8]
        if int(date[0:4]) < year_thresh:
            continue

        print("%d: process %s" % (process_num, sfn))
        # load data
        src = xa.load_dataset(sfn)
        sst = src['sea_surface_temperature'].data
        lat = src['lat'].data
        lon = src['lon'].data

        # NOTE : convert temparature from kelvin to celcius
        # sst = sst - 273.15

        # trimming & resize
        lat_mask = np.logical_and(lat_range[0] < lat, lat < lat_range[1])
        lon_mask = np.logical_and(lon_range[0] < lon, lon < lon_range[1])
        dst = sst[0, lat_mask, :]
        dst = dst[:, lon_mask]

        dst = cv2.resize(dst, map_size)

        # save file
        save_name = os.path.join(sst_save_dir, date + ".npy")
        np.save(save_name, dst)


def preprocess_sea_surface_temparature_data(data_pattern='day'):

    # set directory to store processed data
    sst_day_dst_dir = os.path.join(DATASET_DIR, "sst-day")
    sst_night_dst_dir = os.path.join(DATASET_DIR, "sst-night")

    # make dirs
    os.makedirs(sst_day_dst_dir, exist_ok=True)
    os.makedirs(sst_night_dst_dir, exist_ok=True)

    # get downloaded file names
    source_filenames_day = glob(os.path.join(ORIGINAL_DATA_SOURCE_DIR, "sst", "*_day-*.nc"))
    source_filenames_night = glob(os.path.join(ORIGINAL_DATA_SOURCE_DIR, "sst", "*_night-*.nc"))

    # select day or night data
    if data_pattern == 'day':
        print("Process day data ...\n")
        source_filenames = source_filenames_day
        sst_save_dir = sst_day_dst_dir
    else:
        print("Process night data ...\n")
        source_filenames = source_filenames_night
        sst_save_dir = sst_night_dst_dir

    # split source filenames
    n_file = int(len(source_filenames) / 4)
    source_filenames1 = source_filenames[:n_file]
    source_filenames2 = source_filenames[n_file-2:int(n_file*2)]
    source_filenames3 = source_filenames[int(n_file*2)-2:int(n_file*3)]
    source_filenames4 = source_filenames[int(n_file*3)-2:]

    threads = [mp.Process(target=multi_process, args=(source_filenames1, sst_save_dir, LON_RANGE, LAT_RANGE, MAP_SIZE, YEAR_RANGE[0], 1,)),
               mp.Process(target=multi_process, args=(source_filenames2, sst_save_dir, LON_RANGE, LAT_RANGE, MAP_SIZE, YEAR_RANGE[0], 2,)),
               mp.Process(target=multi_process, args=(source_filenames3, sst_save_dir, LON_RANGE, LAT_RANGE, MAP_SIZE, YEAR_RANGE[0], 3,)),
               mp.Process(target=multi_process, args=(source_filenames4, sst_save_dir, LON_RANGE, LAT_RANGE, MAP_SIZE, YEAR_RANGE[0], 4,))]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("done.")


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['day', 'night'])
    args = parser.parse_args()

    # process data
    preprocess_sea_surface_temparature_data(data_pattern=args.type)
