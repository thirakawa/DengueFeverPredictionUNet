#!/usr/bin/env python3


import os
from glob import glob
import numpy as np
import cv2

from argparse import ArgumentParser

VALUE_RANGE = {
    'dengue_binary': (0.0, 1.0),
    'dengue': (0.0, 30.0),
    'precipitation': (0.0, 121.58144),
    'solar': (0.0, 377.1394),
    'sst': (0.0, 313.13998)
}


def parser():
    arg_parser = ArgumentParser()

    arg_parser.add_argument('--data_dir', '-d', type=str, required=True,
                            help='directory to be saved visualized data')
    arg_parser.add_argument('--output', '-o', type=str, required=True,
                            help='filename for output video')

    # data type
    arg_parser.add_argument('--type', '-t', type=str,
                            choices=['dengue', 'dengue_binary',
                                     'sst-day-interpolated', 'sst-night-interpolated',
                                     'solar', 'precipitation'],
                            help='data type for making video')

    return arg_parser.parse_args()


def main():

    args = parser()

    # get original image filenames
    filenames = glob(os.path.join(args.data_dir, args.type, "*.npy"))
    filenames.sort()

    # get image height and width
    tmp = np.load(filenames[0])
    H, W = tmp.shape

    if args.type == 'sst-day-interpolated' or args.type == 'sst-night-interpolated':
        value_range = VALUE_RANGE['sst']
    else:
        value_range = VALUE_RANGE[args.type]

    # make video writer instance
    print("making video for", args.data_dir)

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.output, codec, 5, (W, H))

    # write
    for fn in filenames:
        data = np.load(fn)
        if args.type == 'dengue_binary':
            data = data.astype(np.uint8) * 255
            img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        else:
            data = data / value_range[1] * 255
            data = data.astype(np.uint8)
            img = cv2.applyColorMap(data.astype(np.uint8), cv2.COLORMAP_JET)

        date = os.path.splitext(os.path.basename(fn))[0]
        img = cv2.putText(img, date, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
        video.write(img)

    # save (release object)
    video.release()


if __name__ == '__main__':
    main()
