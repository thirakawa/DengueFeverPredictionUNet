# Dengue Dataset Pre-Processing Scripts


## Requirements

* Python3.x
  * xarray
  * netcdf4
* aria2c



---
## 1. Download Source Data

First, we need to download source data from each website.
For downloading source data, we should work in `download_source` directory.


### a. Set config file

Please modify `download_config.py` as you need.
In this Python script, please modify `ORIGINAL_SOURCE_DIR` in accordance with your environment.
```python
ORIGINAL_SOURCE_DIR = "/Please/set/path/to/save/directory"
```

Note that I recommend that you DO NOT touch the other parts.
Because the other variables and functions are related to the next preprocessing part.


### b. Download file

Download each source data by executing the following commands.
```bash
python3 download_dengue.py
python3 download_precipitation.py
python3 download_solar.py
python3 download_sea_surface_temparature.py
```

Because a number of source will take a long time to download all data, please check your machine and network settings to keep network connection.
For example, turning off sleep mode and so on.



---
## 2. Pre-Process for Source Data

Second, we process source data to obtain numpy format objects.
To process data, we work in `preprocess` directory.


### a. Set config file

Please modify the following variables in `config.py` as you need.

* `ORIGINAL_DATA_SOURCE_DIR`: path to directory that stores source data. This would be the same path as `ORIGINAL_SOURCE_DIR` in the above download source data script.
* `DATASET_DIR`: path to save processed data.
* `LON_RANGE`: the range of longitude to trim data
* `LAT_RANGE`: the range of latitude to trim data
* `MAP_SIZE`: the 2D array size of trimmed data
* `YEAR_RANGE`: the range of years to process data (you don't have to change it.)


### b. Pre-Process data
Next, we preprocess every data by executing the following commands.

```bash
python3 preprocess_dengue.py
python3 preprocess_precipitation.py
python3 preprocess_solar.py
python3 preprocess_sea_surface_temparature.py --type day
python3 preprocess_sea_surface_temparature.py --type night
```


### c. interpolate SST (Sea Surface Temparature) Data

We interpolate SST data.

```bash
python3 preprocess_sst_interpolation.py --type day
python3 preprocess_sst_interpolation.py --type night
```


### d. Other scripts for checking and interpolating data

The following scripts are used for checking processed data.

* `check_missing_file.py`: Check missing processed data.
* `check_data_value_range.py`: Check min max values of each data



---
## 3. Merge Daily Data

For merging daily data into a specifc days or weeks, please work in `merge_data` directory.


### a. Rewrite merge settings

First, we need to rewirte merge settings (parameters) in `merge_config.py`.

```python
# the file (data) which start to merge data
START_DATE = '20020416'
# directory to pre-merged data
SRC_DIR = "/data/hirakawa/Dengue/DengueDataset/1day"
# (home, base) directory to store merged data
DST_HOME_DIR = "/data/hirakawa/Dengue/DengueDataset"
```


### b. Merge data

Please execute the following commands.

```bash
python3 merge_data_2days.py
python3 merge_data_1week.py
python3 merge_data_2weeks.py
```
