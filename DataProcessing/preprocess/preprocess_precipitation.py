#!/usr/bin/env python3

import os
from glob import glob
import numpy as np
import cv2

from config import *


def load_precipitation_data(input_filename):
    with open(input_filename, 'rb') as f:
        data = np.fromfile(f, np.float32).reshape([1200, 3600])
    return data


def interpolate_missing_precipitation(input_data):
    input_data[input_data <= -999] = 0
    return input_data


def trimming(data):
    x_range = [int(LON_RANGE[0] * 10), int(LON_RANGE[1] * 10)]
    y_range = [int((60 - LAT_RANGE[1]) * 10), int((60 - LAT_RANGE[0]) * 10)]

    dst = data[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    dst = cv2.resize(dst, MAP_SIZE)

    return dst


def preprocess_precipitation():

    # set directory to store processed data
    precipitation_dst_dir = os.path.join(DATASET_DIR, "precipitation")
    
    # make dirs
    os.makedirs(precipitation_dst_dir, exist_ok=True)

    # get downloaded file names
    source_filenames = glob(os.path.join(ORIGINAL_DATA_SOURCE_DIR, "precipitation", "*.dat"))
    source_filenames.sort()

    # process data
    for sfn in source_filenames:
        date = os.path.basename(sfn).split('.')[1]
        if int(date[0:4]) < YEAR_RANGE[0]:
            continue

        print("process %s" % sfn)

        # load data
        src = load_precipitation_data(sfn)

        # interpolate missing value
        # TODO : interpolate missing value (-999.9 --> ???; what is better???)
        dst = interpolate_missing_precipitation(src)

        # trimming
        dst = trimming(dst)

        # save flle
        save_name = os.path.join(precipitation_dst_dir, date + ".npy")
        np.save(save_name, dst)

    print("done.\n\n")


if __name__ == "__main__":
    preprocess_precipitation()
