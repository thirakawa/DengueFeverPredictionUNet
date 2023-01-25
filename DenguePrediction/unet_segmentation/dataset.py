#!/usr/bin/env python3

import sys
from os import path
from random import randint
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from glob import glob
from datetime import date, timedelta
import cv2
from copy import deepcopy
import torch
from torch.utils.data import Dataset


DENGUE_VALUE_RANGE = (0.0, 30.0)  # (0.0, 29.0)
PRECIPITATION_VALUE_RANGE = (0.0, 121.58144)
SOLAR_VALUE_RANGE = (0.0, 377.1394)
SST_VALUE_RANGE = (0.0, 313.13998)


DENGUE_DIR = "dengue"
DENGUE_BIN_DIR = "dengue_binary"
PRECIPITATION_DIR = "precipitation"
SOLAR_DIR = "solar"
SST_DAY_DIR = "sst-day-interpolated"
SST_NIGHT_DIR = "sst-night-interpolated"


DATA_DATE_RANGE = {'train': (date(2002, 4, 16), date(2017, 12, 31)),
                   'val'  : (date(2018, 1, 1),  date(2018, 12, 31)),
                   'test' : (date(2019, 1, 1),  date(2020, 9, 30)),
                   'all'  : (date(2002, 4, 16), date(2020, 9, 30))}


def date_range(start, stop, step=timedelta(1)):
    current = start
    while current < stop:
        yield current
        current += step


class DengueDataset(Dataset):

    def __init__(self, data_dir, mode='train', delay=1, binary_label=False,
                 use_dengue=True, sst_type='day', use_precipitation=True, use_solar=True):
        super().__init__()

        self.sigma = 2.0

        self.data_dir = data_dir
        if mode in ['train', 'val', 'test', 'all']:
            self.mode = mode
        else:
            print("ERROR: invalid data mode:", mode)
            sys.exit(-1)
        self.delay = delay
        self.binary_label = binary_label

        # select data type
        self.use_dengue = use_dengue
        if sst_type in ['day', 'night', 'both', 'none']:
            self.sst_type = sst_type
        else:
            print("ERROR: invalid sst type:", sst_type)
            sys.exit(-1)
        self.use_precipitation = use_precipitation
        self.use_solar = use_solar

        # compute the nubmer of input data channels
        self.n_channels = 0
        if self.use_dengue:
            self.n_channels += 1
        if sst_type == 'day' or sst_type == 'night':
            self.n_channels +=1
        elif sst_type == 'both':
            self.n_channels += 2
        if self.use_precipitation:
            self.n_channels  += 1
        if self.use_solar:
            self.n_channels += 1
        if self.n_channels == 0:
            print("ERROR: please select at least one data...")
            sys.exit(-1)

        self.basename_list = self._get_basename_list()

    def _get_basename_list(self):
        _date_list_tmp = []
        for dd in date_range(DATA_DATE_RANGE[self.mode][0], DATA_DATE_RANGE[self.mode][1]):
            _date_name = dd.strftime("%Y%m%d")
            _date_list_tmp.append(_date_name)
        
        _filenames = glob(path.join(self.data_dir, "dengue", "*.npy"))
        date_list = []
        for d in _date_list_tmp:
            for fn in _filenames:
                if d in fn:
                    date_list.append(d)
                    break

        return date_list

    def _make_gaussian_map(self, base_map):
        if np.sum(base_map > 0.5) == 0:
            return base_map.astype(np.float32)
        else:
            blurred = gaussian_filter(base_map, sigma=self.sigma)
            # normalize
            blurred = (blurred - np.min(blurred)) / (np.max(blurred) - np.min(blurred))
            return blurred

    def _load_label(self, input_basename):
        if self.binary_label:
            _filename = path.join(self.data_dir, DENGUE_BIN_DIR, input_basename + ".npy")
            data = np.load(_filename).astype(np.int64)
        else:
            # _filename = path.join(self.data_dir, DENGUE_DIR, input_basename + ".npy")
            # data = np.load(_filename).astype(np.float32) / DENGUE_VALUE_RANGE[1]
            _filename = path.join(self.data_dir, DENGUE_BIN_DIR, input_basename + ".npy")
            data = np.load(_filename).astype(np.float32)
            data = self._make_gaussian_map(data)
        return data

    def _load_sst(self, input_basename):
        if self.sst_type == 'day':
            _filename = path.join(self.data_dir, SST_DAY_DIR, input_basename + ".npy")
            _data = np.load(_filename)
            _data[_data >= 0] = _data[_data >= 0] / SST_VALUE_RANGE[1]
            return _data, None
        elif self.sst_type == 'night':
            _filename = path.join(self.data_dir, SST_NIGHT_DIR, input_basename + ".npy")
            _data = np.load(_filename)
            _data[_data >= 0] = _data[_data >= 0] / SST_VALUE_RANGE[1]
            return _data, None
        elif self.sst_type == 'both':
            _filename1 = path.join(self.data_dir, SST_DAY_DIR, input_basename + ".npy")
            _filename2 = path.join(self.data_dir, SST_NIGHT_DIR, input_basename + ".npy")
            _data1 = np.load(_filename1)
            _data2 = np.load(_filename2)
            _data1[_data1 >= 0] = _data1[_data1 >= 0] / SST_VALUE_RANGE[1]
            _data2[_data2 >= 0] = _data2[_data2 >= 0] / SST_VALUE_RANGE[1]
            return _data1, _data2

    def _load_data(self, input_basename):
        _data = []

        # dengue
        if self.use_dengue:
            _data.append(self._load_label(input_basename))

        # sst
        if self.sst_type != 'none':
            _d1, _d2 = self._load_sst(input_basename)
            _data.append(_d1)
            if _d2 is not None:
                _data.append(_d2)

        # precipitation
        if self.use_precipitation:
            _filename = path.join(self.data_dir, PRECIPITATION_DIR, input_basename + ".npy")
            _data.append(np.load(_filename) / PRECIPITATION_VALUE_RANGE[1])

        # solar
        if self.use_solar:
            _filename = path.join(self.data_dir, SOLAR_DIR, input_basename + ".npy")
            _data.append(np.load(_filename) / SOLAR_VALUE_RANGE[1])

        return np.asarray(_data).astype(np.float32)

    def __getitem__(self, item):
        _data_basename = self.basename_list[item]
        _label_basename = self.basename_list[item + self.delay]
        _data = self._load_data(_data_basename)
        _label = self._load_label(_label_basename)
        return _data, _label, _data_basename

    def __len__(self):
        return len(self.basename_list) - self.delay


if __name__ == "__main__":

    print("debug.")

    import os
    from time import time
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    DATA_ROOT = "/raid/hirakawa/Dengue/DengueDataset/v2_2week"

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

    _start = time()
    train_data = DengueDataset(data_dir=DATA_ROOT,
                                         mode='train', binary_label=True,
                                         sst_type='night', use_precipitation=False, use_solar=False)
    val_data = DengueDataset(data_dir=DATA_ROOT,
                                         mode='val', binary_label=True,
                                         sst_type='night', use_precipitation=False, use_solar=False)
    test_data = DengueDataset(data_dir=DATA_ROOT,
                                         mode='test', binary_label=True,
                                         sst_type='night', use_precipitation=False, use_solar=False)
    all_data = DengueDataset(data_dir=DATA_ROOT,
                                       mode='all', binary_label=True,
                                       sst_type='day', use_precipitation=True, use_solar=True)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=False, **kwargs)
    for input_data, target_label, basename in train_loader:
        print(type(input_data), type(target_label), input_data.size(), target_label.size(), basename)

    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, **kwargs)
    for input_data, target_label, basename in val_loader:
        print(type(input_data), type(target_label), input_data.size(), target_label.size(), basename)

    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, **kwargs)
    for input_data, target_label, basename in test_loader:
        print(type(input_data), type(target_label), input_data.size(), target_label.size(), basename)

    all_loader = DataLoader(all_data, batch_size=1, shuffle=False, **kwargs)
    for input_data, target_label, basename in all_loader:
        print(type(input_data), type(target_label), input_data.size(), target_label.size(), basename)
