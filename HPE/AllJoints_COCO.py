#!/usr/bin/python3
# coding=utf-8

'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Adapted from: https://github.com/chenyilun95/tf-cpn/blob/master/data/COCO/COCOAllJoints.py
'''
import os
import os.path as osp
import numpy as np
import cv2

import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join('../data/COCO', 'MSCOCO', 'PythonAPI'))

from pycocotools.coco import COCO

class PoseTrackJoints_COCO(object):
    def __init__(self):
        self.kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
        'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
        'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
        self.max_num_joints = 17
        self.color = np.random.randint(0, 256, (self.max_num_joints, 3))

        self.mpi = []
        self.test_mpi = []
        for mpi, stage in zip([self.mpi, self.test_mpi], ['train', 'val']):
            if stage == 'train':
                self._train_gt_path=os.path.join('../data/COCO', 'MSCOCO', 'annotations', 'person_keypoints_trainvalminusminival2014.json')

                coco = COCO(self._train_gt_path)
            else:
                self._val_gt_path=os.path.join('../data/COCO', 'MSCOCO', 'annotations', 'person_keypoints_minival2014.json')

                coco = COCO(self._val_gt_path)
            if stage == 'train':
                for aid in coco.anns.keys():
                    ann = coco.anns[aid]
                    if ann['image_id'] not in coco.imgs or ann['image_id'] == '366379':
                        continue
                    imgname = coco.imgs[ann['image_id']]['file_name']
                    prefix_head = "../data/COCO/MSCOCO/images/"
                    prefix = 'val' if 'val' in imgname else 'train'
                    rect = np.array([0, 0, 1, 1], np.int32)
                    if ann['iscrowd']:
                        continue
                    joints = ann['keypoints']

                    ''' change the COCO order into PoseTrack order'''
                    joints = change_order_COCO_to_PoseTrack(joints)

                    bbox = ann['bbox']
                    if np.sum(joints[2::3]) == 0 or ann['num_keypoints'] == 0 :
                        continue
                    imgname = prefix_head + prefix + '2014/' + 'COCO_' + prefix + '2014' + '_' + str(ann['image_id']).zfill(12) + '.jpg'
                    humanData = dict(aid = aid,joints=joints, imgpath=imgname, headRect=rect, bbox=bbox, imgid = ann['image_id'], segmentation = ann['segmentation'])
                    mpi.append(humanData)
            elif stage == 'val':
                files = [(img_id,coco.imgs[img_id]) for img_id in coco.imgs]
                for img_id, img_info in files:
                    imgname = stage + '2014/' + img_info['file_name']
                    humanData = dict(imgid = img_id,imgpath = imgname)
                    mpi.append(humanData)
            else:
                print('PoseTrack_COCO data error, please check')
                embed()

    def load_data(self, min_kps=1):
        mpi = [i for i in self.mpi if np.sum(np.array(i['joints'], copy=False)[2::3] > 0) >= min_kps]
        return mpi, self.test_mpi


def change_order_COCO_to_PoseTrack(pose_keypoints_2d_COCO):
    #COCO:      {0-nose    1-Leye    2-Reye    3-Lear    4Rear    5-Lsho    6-Rsho    7-Lelb    8-Relb    9-Lwri    10-Rwri    11-Lhip    12-Rhip    13-Lkne    14-Rkne    15-Lank    16-Rank}　
    #Posetrack: {0-Rank    1-Rkne    2-Rhip    3-Lhip    4-Lkne    5-Lank    6-Rwri    7-Relb    8-Rsho    9-Lsho   10-Lelb    11-Lwri    12-neck  13-nose　14-TopHead}

    order_mapping = {0:13, 1:14, 2:14, 3:14, 4:14, 5:9, 7:10, 9:11, 6:8, 8:7, 10:6, 11:3, 13:4, 15:5, 12:2, 14:1, 16:0}

    num_keypoints_COCO = int(len(pose_keypoints_2d_COCO)/3)
    pose_keypoints_2d_PoseTrack = 15*3*[0]

    for index_COCO in range(num_keypoints_COCO):
        x = pose_keypoints_2d_COCO[3*index_COCO]
        y =  pose_keypoints_2d_COCO[3*index_COCO + 1]
        score = pose_keypoints_2d_COCO[3*index_COCO + 2]
        index_PoseTrack = order_mapping[index_COCO]

        if index_PoseTrack == 12:
            continue
        elif index_PoseTrack == 14:
            continue
        else:
            pose_keypoints_2d_PoseTrack[3*index_PoseTrack] = x
            pose_keypoints_2d_PoseTrack[3*index_PoseTrack +1] = y
            pose_keypoints_2d_PoseTrack[3*index_PoseTrack +2] = score

        pose_keypoints_2d_PoseTrack[3*12] = (pose_keypoints_2d_COCO[3*5] + pose_keypoints_2d_COCO[3*6])/2
        pose_keypoints_2d_PoseTrack[3*12+1] = (pose_keypoints_2d_COCO[3*5 +1] + pose_keypoints_2d_COCO[3*6 +1])/2
        pose_keypoints_2d_PoseTrack[3*12+2] = (pose_keypoints_2d_COCO[3*5 +2] + pose_keypoints_2d_COCO[3*6 +2])/2

        pose_keypoints_2d_PoseTrack[3*14] = (pose_keypoints_2d_COCO[3*1] + pose_keypoints_2d_COCO[3*2])/2
        pose_keypoints_2d_PoseTrack[3*14+1] = 2 * pose_keypoints_2d_PoseTrack[3*13+1] - pose_keypoints_2d_PoseTrack[3*12+1]
        pose_keypoints_2d_PoseTrack[3*14+2] = (pose_keypoints_2d_COCO[3*1 +2] + pose_keypoints_2d_COCO[3*2 +2])/2
    return pose_keypoints_2d_PoseTrack


if __name__ == '__main__':
    coco_joints = PoseTrackJoints_COCO()
    train, test = coco_joints.load_data(min_kps=1)
    from IPython import embed; embed()
