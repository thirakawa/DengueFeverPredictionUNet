#!/usr/bin/env python3

import os
from glob import glob
import numpy as np
import cv2

from config import *


def load_solar_data(input_filename):
    with open(input_filename, 'rb') as f:
        data = np.fromfile(f, np.uint8)
    data = data.reshape(-1, 7200)
    data = data.astype(np.float32)
    data[data == 255] = np.nan
    data = data.astype(np.float32) * 1.60 + 0.0
    return data[1:, :]  # remove header

def merge_solar_data(data1, data2):
    nanmap1 = np.isnan(data1)
    nanmap2 = np.isnan(data2)

    both_value_map = np.logical_and(np.logical_not(nanmap1), np.logical_not(nanmap2))
    single_value_map1 = np.logical_and(np.logical_not(both_value_map), np.logical_not(nanmap1))
    single_value_map2 = np.logical_and(np.logical_not(both_value_map), np.logical_not(nanmap2))
    none_value_map = np.logical_and(nanmap1, nanmap2)

    dst = np.zeros(data1.shape, dtype=np.float32)
    dst[both_value_map] = (data1[both_value_map] + data2[both_value_map]) / 2.0
    dst[single_value_map1] = data1[single_value_map1]
    dst[single_value_map2] = data2[single_value_map2]
    dst[none_value_map] = np.nan

    return dst

def interpolate_nan(x):
    '''
    Replace nan by interporation
    Source URL: http://pytan.hatenablog.com/entry/2015/06/29/012706
                http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    '''
    ok = np.logical_not(np.isnan(x))
    xp = ok.nonzero()[0]
    fp = x[ok]
    _x = np.isnan(x).nonzero()[0]
    x[np.logical_not(ok)] = np.interp(_x, xp, fp)
    return x

def trimming(data):
    lon_list = np.linspace(0, 360, 7200)
    lat_list = np.linspace(90, -90, 3601)

    lon_mask = np.logical_and(lon_list >= LON_RANGE[0], lon_list <=LON_RANGE[1])
    lat_mask = np.logical_and(lat_list >= LAT_RANGE[0], lat_list <=LAT_RANGE[1])

    dst = data[lat_mask, :]
    dst = dst[:, lon_mask]
    return dst

def find_file(query, filenames):
    match_index = None
    for i, fn in enumerate(filenames):
        if query in fn:
            match_index = i
            break
    
    if match_index is not None:
        found_filename = filenames.pop(match_index)
        return found_filename, filenames
    else:
        return None, filenames


def preprocess_solar_radiation():

    # set directory to store processed data
    solar_dst_dir = os.path.join(DATASET_DIR, "solar")

    # make dirs
    os.makedirs(solar_dst_dir, exist_ok=True)

    # get downloaded file names
    terra_source_filenames = glob(os.path.join(ORIGINAL_DATA_SOURCE_DIR, "solar", "MOD*__8b"))
    aqua_source_filenames = glob(os.path.join(ORIGINAL_DATA_SOURCE_DIR, "solar", "MYD*__8b"))
    terra_source_filenames.sort()
    aqua_source_filenames.sort()

    # organize terra and aqua set
    modis_set_files = []
    for sfn in terra_source_filenames:
        file_date = os.path.basename(sfn).split('_')[1][1:9]
        aqua_fn, aqua_source_filenames = find_file(file_date, aqua_source_filenames)
        modis_set_files.append([sfn, aqua_fn])
    
    for sfn in aqua_source_filenames:
        file_date = os.path.basename(sfn).split('_')[1][1:9]
        modis_set_files.append([None, sfn])

    # process data
    for terra_fn, aqua_fn in modis_set_files:
        print("process %s and %s" % (terra_fn, aqua_fn))

        if terra_fn is None:
            src = load_solar_data(aqua_fn)
            date = os.path.basename(aqua_fn).split('_')[1][1:9]

        elif aqua_fn is None:
            src = load_solar_data(terra_fn)
            date = os.path.basename(terra_fn).split('_')[1][1:9]

        else:
            src1 = load_solar_data(terra_fn)
            src2 = load_solar_data(aqua_fn)
            src = merge_solar_data(src1, src2)
            date1 = os.path.basename(terra_fn).split('_')[1][1:9]
            date2 = os.path.basename(aqua_fn).split('_')[1][1:9]
            if date1 != date2:
                print("ERROR: %d and %d are different date." % (date1, date2))
                exit(-1)
            date = date1

        if int(date[0:4]) < YEAR_RANGE[0]:
            continue

        # interpolate nan
        dst = interpolate_nan(src)

        # trimming
        dst = trimming(dst)

        # resize
        dst = cv2.resize(dst, MAP_SIZE)

        # save file
        save_name = os.path.join(solar_dst_dir, date + ".npy")
        np.save(save_name, dst)

    print("done.")


if __name__ == "__main__":
    preprocess_solar_radiation()
