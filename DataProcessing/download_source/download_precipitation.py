#!/usr/bin/env python3

import os
from glob import glob
import urllib.request
import ftplib

from download_config import PRECIPITATION_DIR, ungzip


# FTP Server Info. ##############################
FTP_HOSTNAME = 'hokusai.eorc.jaxa.jp'
DIR_PATH = 'standard/v6/daily/00Z-23Z/'
ID = "rainmap"
PASS = "Niskur+1404"
#################################################


def download_precipitation():
    # check existing data
    print("Check existing original files ...")
    existing_files = []
    for file in glob(os.path.join(PRECIPITATION_DIR, "*.dat.gz")):
        existing_files.append(os.path.basename(file))

    # make directory
    os.makedirs(PRECIPITATION_DIR, exist_ok=True)

    # download data
    with ftplib.FTP() as ftp:
        ftp.connect(FTP_HOSTNAME)
        ftp.login(ID, PASS)
        ftp.cwd(DIR_PATH)
        year_dirs = ftp.nlst()
        year_dirs.sort()

        for ydir in year_dirs:
            ftp.cwd(ydir)
            filenames = ftp.nlst()

            for fn in filenames:
                if fn not in existing_files:
                    print("download %s" % fn)
                    with open(os.path.join(PRECIPITATION_DIR, fn), 'wb') as f:
                        ftp.retrbinary('RETR ' + fn, f.write)
                    print("    extract %s" % fn)
                    ungzip(os.path.join(PRECIPITATION_DIR, fn))

            ftp.cwd("..")

    print("Download precipitation data; done.")


if __name__ == "__main__":
    download_precipitation()
