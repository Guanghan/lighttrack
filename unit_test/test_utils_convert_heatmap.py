'''
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''

import sys, os
sys.path.append(os.path.abspath("../utils/"))
from utils_convert_heatmap import *
import numpy as np
import cv2

def test_rebin():
    np_array = np.arange(36).reshape([6, 6])
    print(np_array)

    cropped = rebin(np_array, (2,3))
    print(cropped)


def test_pad_heatmaps():
    heatmap = np.arange(64).reshape([8, 8])
    heatmaps = []
    for n in range(3):
        heatmaps.append(heatmap)

    norm_size = 8
    scale = 0.8
    heatmaps_pad = pad_heatmaps(heatmaps, norm_size, scale)
    print(heatmaps_pad)


def main():
    print("Testing: utils_convert_heatmap")

    passed = test_rebin()
    if passed is False:
        print("\t test_rebin failed")

    passed = test_pad_heatmaps()
    if passed is False:
        print("\t test_pad_heatmaps failed")


if __name__ == '__main__':
    main()
