#!/usr/bin/env python3

import os
from glob import glob
import urllib.request
import ftplib

from download_config import SOLAR_DIR, YEAR_RANGE, ungzip


# JASMES FTP Server Info. #######################
JASMES_FTP_HOSTNAME = "apollo.eorc.jaxa.jp"
DIR_PATH = "pub/JASMES/Global_05km/swr/daily/"
#################################################


def download_solar_radiation():
    # check existing files
    print("Check existing original files ...")
    existing_files = []
    for file in glob(os.path.join(SOLAR_DIR, "*.gz")):
        existing_files.append(os.path.basename(file))

    # make directory
    os.makedirs(SOLAR_DIR, exist_ok=True)

    # download data
    with ftplib.FTP() as ftp:
        ftp.connect(JASMES_FTP_HOSTNAME)
        ftp.login()
        ftp.cwd(DIR_PATH)
        year_month_dirs = ftp.nlst()

        for ymdir in year_month_dirs:
            if int(ymdir[:4]) < YEAR_RANGE[0]:
                continue

            ftp.cwd(ymdir)
            filenames = ftp.nlst()

            for fn in filenames:
                if fn not in existing_files:
                    print("download %s" % fn)
                    with open(os.path.join(SOLAR_DIR, fn), 'wb') as f:
                        ftp.retrbinary('RETR ' + fn, f.write)
                    print("    extract %s" % fn)
                    ungzip(os.path.join(SOLAR_DIR, fn))

            ftp.cwd("..")
    print("Download solar radiation data; done.")


if __name__ == "__main__":
    download_solar_radiation()
