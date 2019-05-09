'''
 Cross-heatmap NMS, designed for heatmap
 A heatmap is a N-by-N 2d matrix

    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''
import cv2
import numpy as np
from collections import namedtuple
#from scipy.stats import threshold

Struct_joint_blob = namedtuple("Struct_joint_blob", "heatmap_id, blob_id, blob_pos, blob_diameter, blob_response")
'''
# For LSP dataset
img_size = 256
num_joints = 14
'''

# For Mask-rcnn dataset
img_size = 56
num_joints = 17
valid_heatmap_ids = [13, 14, 15, 16] # left knee, right knee, left ankle, right ankle

def find_joints_in_heatmaps_nms_selected(heatmaps, joints_from_max):
    joint_blobs_list = find_joint_blobs_list_in_heatmaps(heatmaps)
    joint_blob_list = convert_blobs_list_to_blob_list(joint_blobs_list)
    joint_blob_list = sort_joint_blob_list_by_response(joint_blob_list)

    selected_joint_blob_list = []
    joints = []
    selected_joint_index = []

    while(True):
        if joint_blob_list == []: break
        max_joint_blob = joint_blob_list[0]
        selected_joint_blob_list.append(max_joint_blob)
        selected_joint_index.append(max_joint_blob.heatmap_id)

        joint_blob_list = remove_blobs_from_same_heatmap(max_joint_blob, joint_blob_list)
        joint_blob_list = remove_blobs_within_range(max_joint_blob, joint_blob_list)
        joint_blob_list = remove_max_blob(joint_blob_list)

    for heatmap_id in range(num_joints):
        # For joints that are not on selected heatmaps, use max value
        if heatmap_id not in valid_heatmap_ids:
            [y, x, prob] = joints_from_max[heatmap_id]
            joint = [int(y), int(x), prob]
            joints.append(joint)
            continue

        if heatmap_id not in selected_joint_index:
            temp_joint_blobs = joint_blobs_list[heatmap_id]
            temp_joint_blobs_sorted = sort_joint_blob_list_by_response(temp_joint_blobs)
            [x, y] = temp_joint_blobs_sorted[0].blob_pos
            joint = [int(y), int(x), 0]
        else:
            [x, y] = selected_joint_blob_list[selected_joint_index.index(heatmap_id)].blob_pos
            if joints_from_max[heatmap_id][2] == 0:
                joint = [int(y), int(x), 0]
            else:
                joint = [int(y), int(x), 1]
        joints.append(joint)
    return joints


def remove_blobs_from_same_heatmap(max_joint_blob, joint_blob_list):
    joint_blob_list_copy = list(joint_blob_list)
    remove_index = []
    for joint_blob_index, joint_blob in enumerate(joint_blob_list):
        if joint_blob != max_joint_blob and max_joint_blob.heatmap_id == joint_blob.heatmap_id:
            remove_index.append(joint_blob_index)

    remove_blob_from_joint_blob_list(joint_blob_list_copy, remove_index)
    return joint_blob_list_copy


def remove_blobs_within_range(max_joint_blob, joint_blob_list):
    joint_blob_list_copy = list(joint_blob_list)
    remove_index = []
    for joint_blob_index, joint_blob in enumerate(joint_blob_list):
        if joint_blob != max_joint_blob and is_overlapped(max_joint_blob, joint_blob):
            remove_index.append(joint_blob_index)

    remove_blob_from_joint_blob_list(joint_blob_list_copy, remove_index)
    return joint_blob_list_copy


def remove_max_blob(sorted_joint_blob_list):
    joint_blob_list_copy = list(sorted_joint_blob_list)
    remove_blob_from_joint_blob_list(joint_blob_list_copy, [0])
    return joint_blob_list_copy


def is_overlapped(blob_1, blob_2):
    pt1 = blob_1.blob_pos
    pt2 = blob_2.blob_pos
    diameter_1 = blob_1.blob_diameter
    diameter_2 = blob_2.blob_diameter
    if dist(pt1, pt2) <= (diameter_1 + diameter_2)*0.5:
        return True
    else:
        return False


def remove_blob_from_joint_blob_list(joint_blob_list, joint_blob_index):
    for index in sorted(joint_blob_index, reverse=True):
        del(joint_blob_list[index])
    return joint_blob_list


def sort_joint_blob_list_by_response(joint_blob_list):
    joint_blob_list.sort(key= lambda x: x.blob_response, reverse = False)
    return joint_blob_list


def sort_joint_blob_list_by_heatmapid(joint_blob_list):
    joint_blob_list.sort(key= lambda x: x.heatmap_id, reverse = False)
    return joint_blob_list


def find_joint_blob_list_in_heatmaps(heatmaps):
    joint_blobs_list = find_joint_blobs_list_in_heatmaps(heatmaps)
    joint_blob_list = convert_blobs_list_to_blob_list(joint_blobs_list)
    return joint_blob_list


def find_joint_blobs_list_in_heatmaps(heatmaps):
    joint_blobs_list = []
    for heatmap_id, heatmap in enumerate(heatmaps):
        if heatmap_id == num_joints: break

        # only process on selected heatmaps, if not all
        if heatmap_id not in valid_heatmap_ids: continue

        joint_blobs = find_joint_blobs_in_heatmap(heatmap, heatmap_id)
        joint_blobs_list.append(joint_blobs)
    return joint_blobs_list


def convert_blobs_list_to_blob_list(joint_blobs_list):
    joint_blob_list = []
    for joint_blobs in joint_blobs_list:
        for joint_blob in joint_blobs:
            joint_blob_list.append(joint_blob)
    return joint_blob_list


def find_joint_blobs_in_heatmap(heatmap, heatmap_id = 0):
    heatmap = normalize_heatmap(heatmap)
    blobs = find_blobs_in_heatmap(heatmap, flag_show_blobs = False)

    joint_blobs = []

    for blob_id, blob in enumerate(blobs):
        #print(blob.pt[1], blob.pt[0])
        #joint_blob = Struct_joint_blob(heatmap_id, blob_id, blob.pt, blob.size, heatmap[blob.pt[1]][blob.pt[0]])
        joint_blob = Struct_joint_blob(heatmap_id, blob_id, blob.pt, blob.size, heatmap[int(blob.pt[1])][int(blob.pt[0])])
        joint_blobs.append(joint_blob)

    if blobs == []:
        peak = np.unravel_index(heatmap.argmin(), heatmap.shape)
        joint_blob = Struct_joint_blob(heatmap_id, 0, (peak[1], peak[0]), 0, heatmap.min())
        joint_blobs.append(joint_blob)

    return joint_blobs


def find_blobs_in_heatmap(heatmap, flag_show_blobs = False):
    heatmap = normalize_heatmap(heatmap)
    blobs = detect_blobs(heatmap)
    if flag_show_blobs is True:
        show_blobs_in_heatmap(heatmap, blobs)
    return blobs


def normalize_heatmap(heatmap):
    if (heatmap.dtype != 'uint8'):
        heatmap = convert_heatmap_float_to_uint8(heatmap)
    if (heatmap.shape != [img_size, img_size]):
        heatmap = cv2.resize(heatmap, (img_size, img_size))
    return heatmap


def detect_blobs(heatmap):
    detector = init_blob_detector()
    blobs = detector.detect(heatmap)
    return blobs


def init_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 1
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    #detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def convert_heatmap_float_to_uint8(heatmap):
    heatmap[heatmap < 0] = 0
    heatmap = 255 - (255.0 * heatmap).astype('uint8')
    return heatmap


def show_blobs_in_heatmap(heatmap, blobs):
    heatmap_with_blobs = cv2.drawKeypoints(heatmap, blobs, np.array([]),
                                           (0,0,255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    [i,j] = np.unravel_index(heatmap.argmin(), heatmap.shape)
    cv2.circle(heatmap_with_blobs, (j,i), 3, (0,255,0))
    cv2.imshow("Heatmap Blobs", heatmap_with_blobs)
    cv2.waitKey(0)


def dist(pt1, pt2):
    [x0, y0] = pt1
    [x1, y1] = pt2

    return np.sqrt((x0-x1)**2 + (y0-y1)**2)
