#!/usr/bin/env python3


import os
from glob import glob
import numpy as np
import cv2

from argparse import ArgumentParser


def parser():
    arg_parser = ArgumentParser()

    arg_parser.add_argument('--data_dir', '-d', type=str, required=True,
                            help='directory to be saved visualized data')
    arg_parser.add_argument('--output', '-o', type=str, required=True,
                            help='filename for output video')

    return arg_parser.parse_args()


def main():

    args = parser()

    # get original image filenames
    filenames = glob(os.path.join(args.data_dir, "*.png"))
    filenames.sort()

    # get image height and width
    tmp = cv2.imread(filenames[0], 1)
    H, W, C = tmp.shape

    # make video writer instance
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(args.data_dir, args.output), codec, 5, (W, H))

    # write
    for fn in filenames:
        img = cv2.imread(fn, 1)
        
        date = os.path.splitext(os.path.basename(fn))[0]
        img = cv2.putText(img, date, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
        
        video.write(img)
    
    # save (release object)
    video.release()


if __name__ == '__main__':
    main()
