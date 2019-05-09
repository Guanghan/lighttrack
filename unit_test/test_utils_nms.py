'''
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''

import sys, os
sys.path.append(os.path.abspath("../utils/"))

import numpy as np
from utils_nms import *

def test_find_joint_blobs_in_heatmap():
    heatmaps = np.load('../dataset_custom/heatmap_sample.npy')

    for ct in range(14):
        heatmap = heatmaps[ct]
        expected_blob = find_joint_blobs_in_heatmap(heatmap)

    if expected_blob is not None:
        return True
    else:
        return False


def test_find_joint_blob_list_in_heatmaps():
    heatmaps = np.load('../dataset_custom/heatmap_sample.npy')

    joint_blob_list = find_joint_blob_list_in_heatmaps(heatmaps)
    print_joint_blob_list_nicely(joint_blob_list)

    print('\n')

    joint_blob_list = sort_joint_blob_list_by_response(joint_blob_list)
    print_joint_blob_list_nicely(joint_blob_list)

    print('\n')
    for ct in range(14):
        print(len(joint_blob_list))
        max_joint_blob = joint_blob_list[0]
        joint_blob_list = remove_blobs_from_same_heatmap(max_joint_blob, joint_blob_list)
        joint_blob_list = remove_blobs_within_range(max_joint_blob, joint_blob_list)
        #print_joint_blob_list_nicely(joint_blob_list)

    if joint_blob_list is not None:
        return True
    else:
        return False


def print_joint_blob_list_nicely(joint_blob_list):
    print('\n')
    for joint_blob in joint_blob_list:
        response = joint_blob.blob_response
        print(response)


def main():
    print("Testing: utils_nms")

    finished = test_find_joint_blob_list_in_heatmaps()
    if finished is not True:
        print("test_find_joint_blob_list_in_heatmaps failed")

    finished = test_find_joint_blobs_in_heatmap()
    if finished is not True:
        print("test_find_joint_blobs_in_heatmap failed")


if __name__ == '__main__':
    main()
