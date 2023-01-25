#!/usr/bin/env python3


import os
import urllib.request
from glob import glob
import requests
from bs4 import BeautifulSoup
import subprocess

from download_config import SST_DIR, YEAR_RANGE

#################################################
# SST URL filename
URL_FILENAME = "sst_urls.txt"
# NOAA website info.
NOAA_URL = "https://data.nodc.noaa.gov/pathfinder/Version5.3/L3C/%d/data/"
#################################################


def get_sst_source_urls():
    print("Get SST data file URLs ...")
    download_urls = []

    for yyyy in range(YEAR_RANGE[0], YEAR_RANGE[1] + 1):
        html_source = requests.get(NOAA_URL % yyyy)
        soup = BeautifulSoup(html_source.text, 'html.parser')
        links = [url.get('href') for url in soup.find_all('a')]
        
        for l in links:
            if ".nc" not in l:
                continue
            download_urls.append(os.path.join(NOAA_URL % yyyy, l))

    # write urls to text file
    with open(URL_FILENAME, 'w') as f:
        for du in download_urls:
            f.write(du + "\n")


def download_sst():

    get_sst_source_urls()

    # make directory
    os.makedirs(SST_DIR, exist_ok=True)

    cmd_list = ["aria2c", "-x", "8", "-j", "8", "-i", URL_FILENAME, "-d", SST_DIR]
    subprocess.call(cmd_list, shell=False)


if __name__ == "__main__":
    download_sst()
