#!/usr/bin/env python3

import os
import urllib.request
from download_config import DENGUE_DIR


# Dengue Data Info. #############################
DENGUE_URL = "https://od.cdc.gov.tw/eic/Dengue_Daily.csv"
DENGUE_FILENAME = "Dengue_Daily.csv"
#################################################


def download_dengue():
    print("Download Dengue Data ...")
    os.makedirs(DENGUE_DIR, exist_ok=True)
    save_fileanme = os.path.join(DENGUE_DIR, DENGUE_FILENAME)
    urllib.request.urlretrieve(DENGUE_URL, save_fileanme)
    print("    Done.\n")


if __name__ == "__main__":
    download_dengue()
