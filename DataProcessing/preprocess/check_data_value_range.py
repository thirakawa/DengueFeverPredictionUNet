#!/usr/bin/env python3

import os
from glob import glob
from datetime import date, timedelta
import numpy as np

from config import *


def check_min_max(search_dir):
    print("Check min max values in %s ..." % search_dir)

    min_val = 100000.
    max_val = -100000.
    filenames = glob(os.path.join(search_dir, "*.npy"))

    for fn in filenames:
        data = np.load(fn)
        _min_tmp = np.min(data)
        _max_tmp = np.max(data)
        if _min_tmp < min_val:
            min_val = _min_tmp
        if _max_tmp > max_val:
            max_val = _max_tmp

    print("    min:", min_val, " max:", max_val, "\n\n")


if __name__ == "__main__":
    check_min_max(os.path.join(DATASET_DIR, "dengue"))
    check_min_max(os.path.join(DATASET_DIR, "precipitation"))
    check_min_max(os.path.join(DATASET_DIR, "solar"))
    check_min_max(os.path.join(DATASET_DIR, "sst-day-interpolated"))
    check_min_max(os.path.join(DATASET_DIR, "sst-night-interpolated"))
