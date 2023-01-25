#!/usr/bin/env python3


from os.path import join
import gzip
import shutil



# Database path settings ------------------------
## path to directory to store downloaded sources
ORIGINAL_SOURCE_DIR = "/data/hirakawa/Dengue/DengueOriginalSource"

## directory to store each source data (I recommend you not to touch.)
DENGUE_DIR = join(ORIGINAL_SOURCE_DIR, "dengue")
SOLAR_DIR = join(ORIGINAL_SOURCE_DIR, "solar")
PRECIPITATION_DIR = join(ORIGINAL_SOURCE_DIR, "precipitation")
SST_DIR = join(ORIGINAL_SOURCE_DIR, "sst")
# -----------------------------------------------



# year range ------------------------------------
YEAR_RANGE = [1998, 2022]
# -----------------------------------------------



# utility functions -----------------------------
## Do not modify
def ungzip(gz_filename):
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(gz_filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
# -----------------------------------------------
