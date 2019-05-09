#!/usr/bin/python3
# coding=utf-8

'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
'''
import os
import os.path as osp
import numpy as np
import cv2

import sys
cur_dir = os.path.dirname(__file__)

from utils_json import read_json_from_file

class PoseTrackJoints(object):
    def __init__(self):
        #{0-Rank    1-Rkne    2-Rhip    3-Lhip    4-Lkne    5-Lank    6-Rwri    7-Relb    8-Rsho    9-Lsho   10-Lelb    11-Lwri    12-neck  13-nose　14-TopHead}
        self.kp_names = ['right_ankle', 'right_knee', 'right_pelvis',
                         'left_pelvis', 'left_knee', 'left_ankle',
                         'right_wrist', 'right_elbow', 'right_shoulder',
                         'left_shoulder', 'left_elbow', 'left_wrist',
                         'upper_neck', 'nose', 'head']
        self.max_num_joints = 15
        self.color = np.random.randint(0, 256, (self.max_num_joints, 3))

        self.posetrack = []
        self.test_posetrack = []
        for posetrack, stage in zip([self.posetrack, self.test_posetrack], ['train', 'val']):
            if stage == 'train':
                self._train_gt_path = "posetrack_merged_train.json"
                gt_python_data = read_json_from_file(self._train_gt_path)
                anns = gt_python_data["annolist"]
            else:
                self._val_gt_path = "posetrack_merged_val.json"
                gt_python_data = read_json_from_file(self._val_gt_path)
                anns = gt_python_data["annolist"]

            if stage == 'train':
                for aid, ann in enumerate(anns):
                    if ann["is_labeled"][0] == 0: continue
                    if not ann["annorect"]:  #if it is empty
                        continue

                    num_candidates = len(ann["annorect"])
                    for candidate_id in range(0, num_candidates):
                        if not ann["annorect"][candidate_id]["annopoints"]: continue  #list is empty

                        # (1) bbox
                        bbox = get_bbox_from_keypoints(ann["annorect"][candidate_id]["annopoints"][0]["point"])
                        bbox = x1y1x2y2_to_xywh(bbox)
                        if bbox == [0, 0, 2, 2]: continue

                        # (2) imgpath
                        imgname = ann["image"][0]["name"]
                        prefix = '../data/posetrack_data/'
                        imgpath = os.path.join(prefix, imgname)

                        # (2) joints
                        joints = get_joints_from_ann(ann["annorect"][candidate_id]["annopoints"][0]["point"])
                        num_points = len(ann["annorect"][candidate_id]["annopoints"][0]["point"])
                        if np.sum(joints[2::3]) == 0 or num_points== 0:
                            continue

                        # (4) head_rect: useless
                        rect = np.array([0, 0, 1, 1], np.int32)

                        ''' This [humanData] is what [load_data] will provide '''
                        humanData = dict(aid = aid, joints=joints, imgpath=imgpath, headRect=rect, bbox=bbox, imgid = ann['imgnum'][0])

                        posetrack.append(humanData)
            elif stage == 'val':
                for aid, ann in enumerate(anns):
                    if ann["is_labeled"][0] == 0: continue
                    if not ann["annorect"]:  #if it is empty
                        continue

                    num_candidates = len(ann["annorect"])
                    for candidate_id in range(0, num_candidates):
                        if not ann["annorect"][candidate_id]["annopoints"]: continue  #list is empty

                        imgname = ann["image"][0]["name"]
                        prefix = '../data/posetrack_data/'
                        imgpath = os.path.join(prefix, imgname)

                        humanData = dict(imgid = aid, imgpath = imgpath)
                        posetrack.append(humanData)
            else:
                print('PoseTrack data error, please check')
                embed()

    def load_data(self, min_kps=1):
        posetrack = [i for i in self.posetrack if np.sum(np.array(i['joints'], copy=False)[2::3] > 0) >= min_kps]
        return posetrack, self.test_posetrack


def get_joints_from_ann(keypoints_python_data):
    num_keypoints = len(keypoints_python_data)
    keypoints_dict = {}
    for pid in range(num_keypoints):
        keypoint_id = keypoints_python_data[pid]["id"][0]
        x = int(keypoints_python_data[pid]["x"][0])
        y = int(keypoints_python_data[pid]["y"][0])
        vis = int(keypoints_python_data[pid]["is_visible"][0]) + 1

        keypoints_dict[keypoint_id] = [x, y, vis]

    for i in range(15):
        if i not in keypoints_dict.keys():
            keypoints_dict[i] = [0, 0, 0]  #Should we set them to zero? Yes! COCO dataset did this too.

    keypoints_list = []
    for i in range(15):
        keypoints = keypoints_dict[i]
        keypoints_list.append(keypoints[0])
        keypoints_list.append(keypoints[1])
        keypoints_list.append(keypoints[2])
    return keypoints_list


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def get_bbox_from_keypoints(keypoints_python_data):
    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(num_keypoints):
        x = keypoints_python_data[keypoint_id]["x"][0]
        y = keypoints_python_data[keypoint_id]["y"][0]
        x_list.append(x)
        y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    scale = 0.2  # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    return bbox


def enlarge_bbox(bbox, scale):
    assert(scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


if __name__ == '__main__':
    joints = PoseTrackJoints()
    train, test = joints.load_data(min_kps=1)
    from IPython import embed; embed()
