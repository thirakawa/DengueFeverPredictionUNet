#!/usr/bin/env python3

import os
from glob import glob
import numpy as np
import multiprocessing as mp

from merge_config import *

# Please change here as you want ##########################
MERGE_DAYS = 2
DST_DIR = os.path.join(DST_HOME_DIR, "2days")
###########################################################


def data_binary(input_data_list):
    H, W = input_data_list[0].shape
    data_type = input_data_list[0].dtype
    dst = np.zeros([H, W], dtype=int)
    for data in input_data_list:
        dst += data
    dst[dst == 0] = 0
    dst[dst != 0] = 1
    dst = dst.astype(np.bool)
    return dst


def data_sum(input_data_list):
    H, W = input_data_list[0].shape
    data_type = input_data_list[0].dtype
    dst = np.zeros([H, W], dtype=data_type)
    for data in input_data_list:
        dst += data
    return dst


def data_mean(input_data_list):
    dst = data_sum(input_data_list)
    dst /= len(input_data_list)
    return dst


def merge_data(src_dir, dst_dir, process_type):
    # make dst data dir
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    # get filenames and the number of files
    filenames = glob(os.path.join(src_dir, "*.npy"))
    filenames.sort()
    n_data = len(filenames)

    # find start index:
    start_index = None
    for i, fn in enumerate(filenames):
        if START_DATE in fn:
            start_index = i
            break
    if start_index is None:
        print("ERROR: cannot find start index file:", START_DATE)
        exit(-1)

    # process each data (merge)
    for i in range(start_index, n_data, MERGE_DAYS):

        if i + MERGE_DAYS > n_data:
            break

        print("merge data: from %s to %s" %(filenames[i], filenames[i + MERGE_DAYS - 1]))

        # collect data
        data_list = []
        for j in range(MERGE_DAYS):
            try:
                data_list.append(np.load(filenames[i+j]))
            except:
                print("ERROR: cannot find:", filenames[i+j])
                exit(-1)
        
        # process data
        if process_type == 'bin':
            merged_data = data_binary(data_list)
        elif process_type == 'sum':
            merged_data = data_sum(data_list)
        elif process_type == 'mean':
            merged_data = data_mean(data_list)
        else:
            print("ERROR: unknown process type:", process_type)
            exit(-1)
        
        # save data
        save_basename = os.path.basename(filenames[i])
        np.save(os.path.join(dst_dir, save_basename), merged_data)

    print("Merge data; Done.")


def main():

    threads = []

    # dengue
    threads.append(
        mp.Process(target=merge_data,
                   args=(os.path.join(SRC_DIR, "dengue"),
                         os.path.join(DST_DIR, "dengue"),
                         'sum',))
    )

    # dengue binary
    threads.append(
        mp.Process(target=merge_data,
                   args=(os.path.join(SRC_DIR, "dengue_binary"),
                         os.path.join(DST_DIR, "dengue_binary"),
                         'bin',))
    )

    # precipitation
    threads.append(
        mp.Process(target=merge_data,
                   args=(os.path.join(SRC_DIR, "precipitation"),
                         os.path.join(DST_DIR, "precipitation"),
                         'mean',))
    )

    # solar
    threads.append(
        mp.Process(target=merge_data,
                   args=(os.path.join(SRC_DIR, "solar"),
                         os.path.join(DST_DIR, "solar"),
                         'mean',))
    )

    # sst (day, interpolated)
    threads.append(
        mp.Process(target=merge_data,
                   args=(os.path.join(SRC_DIR, "sst-day-interpolated"),
                         os.path.join(DST_DIR, "sst-day-interpolated"),
                         'mean',))
    )

    # sst (night, interpolated)
    threads.append(
        mp.Process(target=merge_data,
                   args=(os.path.join(SRC_DIR, "sst-night-interpolated"),
                         os.path.join(DST_DIR, "sst-night-interpolated"),
                         'mean',))
    )

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print("Merge data; done.")


if __name__ == '__main__':
    main()
