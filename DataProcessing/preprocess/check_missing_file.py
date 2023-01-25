#!/usr/bin/env python3

import os
from glob import glob
from datetime import date, timedelta
import numpy as np

from config import *


def search_missing_files(search_dir, start_date, end_date, output_filename=None):
    missing_date = []

    for dd in date_range(start_date, end_date):
        date_name = dd.strftime("%Y%m%d")
        if not os.path.exists(os.path.join(search_dir, date_name + ".npy")):
            missing_date.append(date_name)

    if output_filename is not None:
        with open(output_filename, 'w') as f:
            for md in missing_date:
                f.write(md + "\n")

    return missing_date


def main():

    s_date = date(2002, 4, 16)
    e_date = date(2020, 9, 30)

    # search missing files
    _ = search_missing_files(os.path.join(DATASET_DIR, "dengue"),
                             s_date, e_date, "missing_dengue.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "dengue_binary"),
                             s_date, e_date, "missing_dengue_bin.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "precipitation"),
                             s_date, e_date, "missing_precipitation.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "solar"),
                             s_date, e_date, "missing_solar.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "sst-day"),
                             s_date, e_date, "missing_sst-day.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "sst-night"),
                             s_date, e_date, "missing_sst-night.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "sst-day-interpolated"),
                             s_date, e_date, "missing_sst-day-interpolated.txt")

    _ = search_missing_files(os.path.join(DATASET_DIR, "sst-night-interpolated"),
                             s_date, e_date, "missing_sst-night-interpolated.txt")


if __name__ == "__main__":
    main()
