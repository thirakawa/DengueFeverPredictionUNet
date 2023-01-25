#!/usr/bin/env python3


from datetime import date, timedelta


# path to original data source directory --------
ORIGINAL_DATA_SOURCE_DIR = "/data/hirakawa/Dengue/DengueOriginalSource"
# -----------------------------------------------



# original mod settings -------------------------
# processed dataset directory
DATASET_DIR = "/data/hirakawa/Dengue/DengueDataset/1day"
# logitude latitude range
LON_RANGE = [119.5, 122.5]  # width 3
LAT_RANGE = [21.5, 25.5]    # width 4
# map size
MAP_SIZE = (384, 512)       # (x, y)
# year range
YEAR_RANGE = [1998, 2021]
# -----------------------------------------------



# utility functions -----------------------------
def date_range(start, stop, step=timedelta(1)):
    current = start
    while current < stop:
        yield current
        current += step
# -----------------------------------------------
