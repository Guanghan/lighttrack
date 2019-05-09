'''
    utils_heatmaps:
                    utilities for generating heatmaps
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Nov 21th, 2018
'''
import numpy as np
import math
import cv2
from keypoint_visualizer import *


global blob_size
blob_size = 20
#blob_size = 10

global heatmap_size
heatmap_size = 56
#heatmap_size = 28

def generate_base_blob(sigma):
    height = width = blob_size

    heatmap = np.zeros((height, width), dtype = np.float32)
    start = 0
    x = height / 2.
    y = width / 2.
    for h in range(height):
        for w in range(width):
            xx = start + w
            yy = start + h
            dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
            if dis > 4.6052:
                continue
            heatmap[h][w] += math.exp(-dis)
            if heatmap[h][w] > 1:
                heatmap[h][w] = 1

    draw_heatmap(heatmap, joint_name="base")
    return heatmap.astype(np.float32)


def test_generate_base_blob():
    sigma = 4.0
    heatmap_base_blob = generate_base_blob(sigma)

    heatmap_visulize = (255*heatmap_base_blob).astype(int)
    filename = "base_blob.png"
    cv2.imwrite(filename, heatmap_visulize)
    return


def save_base_blob(base_blob):
    np.save("temp_base_blob", base_blob)
    return

def load_base_blob():
    base_blob_path = "temp_base_blob.npy"
    base_blob = np.load(base_blob_path)
    return base_blob


def generate_heatmaps_for_batch(keypoints_batch, base_blob):
    '''
    For a batch of normalized rois, generate a set of heatmaps for each of them

    Input:
        keypoints_batch: (candidate, (x, y, vis), channel)
        base_blob: the gaussian blob centered in the small image

    Output:
        heatmaps_all: (num_candidates, num_channels, heatmap_size, heatmap_size)
    '''
    num_candidates = len(keypoints_batch)
    #num_channels = len(keypoints_batch[0][0][0])
    num_channels = len(keypoints_batch[0][0]) + 1
    #print("num_candidates", num_candidates)
    #print("num_channels", num_channels)

    heatmaps_all = np.zeros((num_candidates, num_channels, heatmap_size, heatmap_size), dtype = np.float32)

    for candidate_id in range(num_candidates):
        # heatmaps: (num_channels, heatmap_size, heatmap_size)
        heatmaps = heatmaps_all[candidate_id]
        #print("Dimension of heatmaps: ", heatmaps.shape)

        for channel_id in range(num_channels):
            heatmap = heatmaps_all[candidate_id, channel_id, :, :]
            #print("Dimension of heatmap: ", heatmap.shape)

            #if True:
            if channel_id != num_channels - 1:
                #print(keypoints_batch)
                keypoint = keypoints_batch[candidate_id, :, channel_id]
                #print("Dimension of keypoint: ", keypoint.shape)

                heatmap = generate_heatmap_for_channel(keypoint, heatmap, base_blob)
            else:
                # generate heatmap for the background
                heatmaps[(num_channels -1), :,:] = 1.0 - np.max(heatmaps[0:(num_channels-1), :,:], axis=0)
    return heatmaps_all


def generate_heatmap_for_channel(keypoint, heatmap, base_blob):
    '''
    Input:
        keypoint: (x, y, vis) in heatmap coordinate system
        heatmap: (heatmap_size, heatmap_size)
        base_blob: the gaussian blob centered in the small image

    Output:
        heatmap: (heatmap_size, heatmap_size)
    '''
    x, y, visibility = keypoint
    '''
    x = max(x, 0)
    y = max(y, 0)
    x = min(x, len(heatmap))
    y = min(y, len(heatmap))
    '''

    offset_x = x - int(blob_size / 2.)
    offset_y = y - int(blob_size / 2.)

    if visibility <= 0:  # 0 means not in image, 1 means exisits but not visible, 2 means exists and visible
        return heatmap
    else:
        if offset_x < 0:
            # need to start from 0
            blob_x_st = abs(offset_x)
        else:
            blob_x_st = 0
        x_st = max(offset_x, 0)

        if offset_y < 0:
            # need to start from 0
            blob_y_st = abs(offset_y)
        else:
            blob_y_st = 0
        y_st = max(offset_y, 0)

        if offset_x + blob_size >= heatmap_size:
            blob_x_end = heatmap_size - x_st
        else:
            blob_x_end = blob_size
        x_end = min(offset_x + blob_size, heatmap_size)

        if offset_y + blob_size >= heatmap_size:
            blob_y_end = heatmap_size - y_st
        else:
            blob_y_end = blob_size
        y_end = min(offset_y + blob_size, heatmap_size)

        if x_end <= 0 or y_end <= 0 or blob_x_end <= 0 or blob_y_end <= 0:
            x_st = 0
            x_end = 0
            blob_x_st = 0
            blob_x_end = 0
            y_st = 0
            y_end = 0
            blob_y_st = 0
            blob_y_end = 0
        #else:
        heatmap[y_st:y_end, x_st:x_end] = base_blob[blob_y_st:blob_y_end, blob_x_st:blob_x_end]
        return heatmap


def test_generate_heatmap_for_channel():
    sigma = 4.0
    base_blob = generate_base_blob(sigma)
    save_base_blob(base_blob)
    loaded_base_blob = load_base_blob()

    keypoint = [56, 56, 2]
    heatmap = np.zeros((56, 56), dtype = np.float32)
    heatmap = generate_heatmap_for_channel(keypoint, heatmap, loaded_base_blob)

    heatmap_visulize = (255*heatmap).astype(int)
    filename = "test_generate_heatmap_for_channel.png"
    cv2.imwrite(filename, heatmap_visulize)
    return


def test_generate_heatmaps_for_batch():
    #sigma = 4.0
    sigma = 3.0
    #sigma = 2.0
    #sigma = 1.
    base_blob = generate_base_blob(sigma)
    save_base_blob(base_blob)
    loaded_base_blob = load_base_blob()

    keypoints_batch = np.array([[[0, 56, 10, 40, 25, 200, -200], [0, 56, 20, 20, 25, 200, -200], [2, 2, 1, 1, 2, 2, 2]]])  # (1, (x, y, vis)=3, channels=4)
    heatmaps_all = generate_heatmaps_for_batch(keypoints_batch, loaded_base_blob)

    num_candidates = len(heatmaps_all)
    for candidate_id in range(num_candidates):
        heatmaps = heatmaps_all[candidate_id]

        demo_heatmaps(heatmaps.astype(float), joint_names)

        num_channels = len(heatmaps)
        for channel_id in range(num_channels):
            heatmap = heatmaps[channel_id]
            print(heatmap)

            heatmap_visulize = (255*heatmap).astype(int)
            filename = "test_{}_{}.png".format(str(candidate_id), str(channel_id))
            cv2.imwrite(filename, heatmap_visulize)
    return



def generate_tilted_heatmaps_for_batch(keypoints_batch, sampled_fg_rois, base_blob):
    '''
    For a batch of non-normalized rois, generate a set of heatmaps for each of them

    Input:
        keypoints_batch: (candidate, (x, y, vis), channel)
        sampled_fg_rois: (candidate, (x1, y1, x2, y2))
        base_blob: the gaussian blob centered in the small image

    Output:
        heatmaps_all: (num_candidates, num_channels, heatmap_size, heatmap_size)
    '''
    num_candidates = len(keypoints_batch)
    #num_channels = len(keypoints_batch[0][0][0])
    num_channels = len(keypoints_batch[0][0]) + 1
    #print("num_candidates", num_candidates)
    #print("num_channels", num_channels)

    heatmaps_all = np.zeros((num_candidates, num_channels, heatmap_size, heatmap_size), dtype = np.float32)

    for candidate_id in range(num_candidates):
        # heatmaps: (num_channels, heatmap_size, heatmap_size)
        heatmaps = heatmaps_all[candidate_id]
        #print("Dimension of heatmaps: ", heatmaps.shape)
        roi = sampled_fg_rois[candidate_id]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = roi

        for channel_id in range(num_channels):
            heatmap = heatmaps_all[candidate_id, channel_id, :, :]
            #print("Dimension of heatmap: ", heatmap.shape)

            #if True:
            if channel_id != num_channels - 1:
                #print(keypoints_batch)
                keypoint = keypoints_batch[candidate_id, :, channel_id]
                #print("Dimension of keypoint: ", keypoint.shape)

                heatmap = generate_tilted_heatmap_for_channel(keypoint, roi, heatmap, base_blob)
            else:
                # generate heatmap for the background
                heatmaps[(num_channels -1), :,:] = 1.0 - np.max(heatmaps[0:(num_channels-1), :,:], axis=0)
    return heatmaps_all


def generate_tilted_heatmap_for_channel(keypoint, roi, heatmap, base_blob):
    '''
    Input:
        keypoint: (x, y, vis) in image coordinate system
        roi: (x1, y1, x2, y2) in image coordinate system
        heatmap: (heatmap_size, heatmap_size)
        base_blob: the gaussian blob centered in the small image

    Output:
        heatmap: (heatmap_size, heatmap_size)
    '''
    keypoint_in_roi = [0, 0, 0]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = roi

    keypoint_in_roi[0] = int(keypoint[0] - bbox_x1)
    keypoint_in_roi[1] = int(keypoint[1] - bbox_y1)
    keypoint_in_roi[2] = keypoint[2]

    roi_wid = int(bbox_x2 - bbox_x1)
    roi_ht = int(bbox_y2 - bbox_y1)
    #print("roi_wid, roi_ht = ", roi_wid, roi_ht)

    roi_heatmap = np.zeros((roi_ht, roi_wid), dtype = np.float32)
    heatmap_size_x, heatmap_size_y = roi_wid, roi_ht

    ''' Generate a heatmap of roi_heatmap size'''
    x, y, visibility = keypoint_in_roi
    offset_x = x - int(blob_size / 2.)
    offset_y = y - int(blob_size / 2.)

    if visibility <= 0:  # 0 means not in image, 1 means exisits but not visible, 2 means exists and visible
        normalized_heatmap = np.zeros((heatmap_size, heatmap_size), dtype = np.float32)
        return normalized_heatmap
    else:
        if offset_x < 0:
            # need to start from 0
            blob_x_st = abs(offset_x)
        else:
            blob_x_st = 0
        x_st = max(offset_x, 0)

        if offset_y < 0:
            # need to start from 0
            blob_y_st = abs(offset_y)
        else:
            blob_y_st = 0
        y_st = max(offset_y, 0)

        if offset_x + blob_size >= heatmap_size_x:
            blob_x_end = heatmap_size_x - x_st
        else:
            blob_x_end = blob_size
        x_end = min(offset_x + blob_size, heatmap_size_x)

        if offset_y + blob_size >= heatmap_size_y:
            blob_y_end = heatmap_size_y - y_st
        else:
            blob_y_end = blob_size
        y_end = min(offset_y + blob_size, heatmap_size_y)

        if x_end <= 0 or y_end <= 0 or blob_x_end <= 0 or blob_y_end <= 0:
            x_st = 0
            x_end = 0
            blob_x_st = 0
            blob_x_end = 0
            y_st = 0
            y_end = 0
            blob_y_st = 0
            blob_y_end = 0
        #if blob_x_end - blob_x_st != x_end - x_st:
        print("blob_x_end, blob_x_st, x_end, x_st = {},{},{},{}".format(blob_x_end, blob_x_st, x_end, x_st))
        print("keypoint, roi = {}, {}".format(keypoint, roi))
        #if blob_y_end - blob_y_st != y_end - y_st:
        print("blob_y_end, blob_y_st, y_end, y_st = {},{},{},{}".format(blob_y_end, blob_y_st, y_end, y_st))
        print("keypoint, roi = {}, {}".format(keypoint, roi))
        roi_heatmap[y_st:y_end, x_st:x_end] = base_blob[blob_y_st:blob_y_end, blob_x_st:blob_x_end]

    ''' Generate a heatmap of normalized size'''
    normalized_heatmap = cv2.resize(roi_heatmap, (heatmap_size, heatmap_size))
    #normalized_heatmap = cv2.resize(roi_heatmap, (heatmap_size, heatmap_size), interpolation=cv2.INTER_NEAREST)
    heatmap[:, :] = normalized_heatmap[:, :]
    return heatmap


def test_generate_tilted_heatmaps_for_batch():
    #sigma = 4.0
    sigma = 2.0
    base_blob = generate_base_blob(sigma)
    save_base_blob(base_blob)
    loaded_base_blob = load_base_blob()

    keypoints_batch = np.array([[[0, 56, 10, 40, 25, 200, -200, 6], [0, 56, 20, 20, 25, 200, -200, 199], [2, 2, 1, 1, 2, 2, 2, 2]]])  # (1, (x, y, vis)=3, channels=4)
    rois = np.array([[25.2, 0, 40, 56], [0, 118.44795, 16.126219, 302.04416]])
    heatmaps_all = generate_tilted_heatmaps_for_batch(keypoints_batch, rois, loaded_base_blob)


    num_candidates = len(heatmaps_all)
    for candidate_id in range(num_candidates):
        heatmaps = heatmaps_all[candidate_id]
        demo_heatmaps(heatmaps, joint_names)

        num_channels = len(heatmaps)
        for channel_id in range(num_channels):
            heatmap = heatmaps[channel_id]

            heatmap_visulize = (255*heatmap).astype(int)
            filename = "test_tilted_{}_{}.png".format(str(candidate_id), str(channel_id))
            cv2.imwrite(filename, heatmap_visulize)
    return


if __name__ == "__main__":
    #test_generate_base_blob()
    #test_generate_heatmap_for_channel()
    test_generate_heatmaps_for_batch()

    #test_generate_tilted_heatmaps_for_batch()
