#!/usr/bin/env python3

import os
from glob import glob
from datetime import date, timedelta
import numpy as np
import multiprocessing as mp

from config import *


def get_date_list(start_date, end_date):
    date_list = []
    for dd in date_range(start_date, end_date):
        date_name = dd.strftime("%Y%m%d")
        date_list.append(date_name)
    return date_list


def median_nan(src, nanmap, window_size=40):
    src_nan = np.isnan(src)
    interpolated_nan = np.logical_xor(src_nan, nanmap)
    yy, xx = np.where(interpolated_nan)

    dst = src.copy()

    for y, x in zip(yy, xx):
        y_start = y - window_size
        y_end = y + window_size
        x_start = x - window_size
        x_end = x + window_size
        if y_start < 0:
            y_start = 0
        if x_start < 0:
            x_start = 0
        crop = src[y_start:y_end, x_start:x_end]
        dst[y, x] = np.nanmedian(crop)
    
    return dst


def check_nan(search_dir):
    filenames = glob(os.path.join(search_dir, "*.npy"))
    for fn in filenames:
        data = np.load(fn)
        if np.sum(np.isnan(data)) != 0:
            print(fn)


def fill_nan(filenames, dst_dir, nan_map_filename):
    nan_region = np.load(nan_map_filename)
    for fn in filenames:
        print("  process:", fn)
        sst_data = np.load(fn)
        interpolated = median_nan(sst_data, nan_region)
        interpolated[nan_region] = -1.0
        interpolated[np.isnan(interpolated)] = -1.0
        basename = os.path.basename(fn)
        np.save(os.path.join(dst_dir, basename), interpolated)


def interpolate_sst(data_pattern='day'):

    # directory path
    SST_DAY_DATASET_DIR = os.path.join(DATASET_DIR, "sst-day")
    SST_NIGHT_DATASET_DIR = os.path.join(DATASET_DIR, "sst-night")
    SST_DAY_DST_DIR = os.path.join(DATASET_DIR, "sst-day-interpolated")
    SST_NIGHT_DST_DIR = os.path.join(DATASET_DIR, "sst-night-interpolated")

    # make directory
    os.makedirs(SST_DAY_DST_DIR, exist_ok=True)
    os.makedirs(SST_NIGHT_DST_DIR, exist_ok=True)

    # set day or night data
    if data_pattern == 'day':
        dataset_dir = SST_DAY_DATASET_DIR
        dst_dir = SST_DAY_DST_DIR
    else:
        dataset_dir = SST_NIGHT_DATASET_DIR
        dst_dir = SST_NIGHT_DST_DIR

    # get sst filenames
    filenames = glob(os.path.join(dataset_dir, "*.npy"))
    filenames.sort()

    # finding common nan region
    print("finding common NaN region ...")
    nan_region = np.ones((MAP_SIZE[1], MAP_SIZE[0]), dtype=np.bool)
    for fn in filenames:
        sst_data = np.load(fn)
        nan_region[np.logical_not(np.isnan(sst_data))] = False
    if data_pattern == 'day':
        nan_map_filename = os.path.join(DATASET_DIR, "nan_map-day.npy")
    else:
        nan_map_filename = os.path.join(DATASET_DIR, "nan_map-night.npy")
    np.save(nan_map_filename, nan_region)

    # interpolate nan region (median)
    print("interpolate nan values ...")

    # for fn in filenames:
    #     print("  process:", fn)
    #     sst_data = np.load(fn)
    #     interpolated = median_nan(sst_data, nan_region)
    #     interpolated[nan_region] = -1.0
    #     interpolated[np.isnan(interpolated)] = -1.0
    #     basename = os.path.basename(fn)
    #     np.save(os.path.join(dst_dir, basename), interpolated)

    # split source filenames
    n_file = int(len(filenames) / 4)
    source_filenames1 = filenames[:n_file]
    source_filenames2 = filenames[n_file-2:int(n_file*2)]
    source_filenames3 = filenames[int(n_file*2)-2:int(n_file*3)]
    source_filenames4 = filenames[int(n_file*3)-2:]
    threads = [mp.Process(target=fill_nan, args=(source_filenames1, dst_dir, nan_map_filename,)),
               mp.Process(target=fill_nan, args=(source_filenames2, dst_dir, nan_map_filename,)),
               mp.Process(target=fill_nan, args=(source_filenames3, dst_dir, nan_map_filename,)),
               mp.Process(target=fill_nan, args=(source_filenames4, dst_dir, nan_map_filename,))]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # interpolate missing file
    print("interpolate missing files ...")
    basename_list = []
    for fn in filenames:
        basename_list.append(os.path.basename(fn))

    s_date = date(2002, 4, 16)
    e_date = date(2020, 9, 30)
    date_list = get_date_list(s_date, e_date)

    for index, dl in enumerate(date_list):
        if dl + ".npy" in basename_list:
            continue

        # backward search
        for b_index in range(index - 1, index - 20, -1):
            if date_list[b_index] + ".npy" in basename_list:
                backward_file = date_list[b_index] + ".npy"
                break

        # foreard search
        for f_index in range(index + 1, index + 20, 1):
            if date_list[f_index] + ".npy" in basename_list:
                forward_file = date_list[f_index] + ".npy"
                break

        b_data = np.load(os.path.join(dst_dir, backward_file))
        f_data = np.load(os.path.join(dst_dir, forward_file))

        averaged = (b_data + f_data) / 2.0
        np.save(os.path.join(dst_dir, dl + ".npy"), averaged)
    print("interplate missing file; done.")

    # check nan values for interpolated data
    print("Check nan values for the interpolated data ...")
    check_nan(dst_dir)
    print("Interpolationl done.\n\n")


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['day', 'night'])
    args = parser.parse_args()

    interpolate_sst(data_pattern=args.type)
