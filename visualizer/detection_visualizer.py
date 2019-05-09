'''
 detection_visualizer.py
 Visualizer for Candidate Detection
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on June 18th, 2018
'''
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from random import random as rand

import os
import cv2
import numpy as np
from utils_io_file import is_image
from utils_io_folder import create_folder
from utils_json import read_json_from_file

bbox_thresh = 0.4

# set up class names for COCO
num_classes = 81  # 80 classes + background class
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def show_boxes_from_python_data(img, dets, classes, output_img_path, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(img)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    plt.savefig(output_img_path)
    return img


def find_color_scalar(color_string):
    color_dict = {
        'purple': (255, 0, 255),
        'yellow': (0, 255, 255),
        'blue':   (255, 0, 0),
        'green':  (0, 255, 0),
        'red':    (0, 0, 255),
        'skyblue':(235,206,135),
        'navyblue': (128, 0, 0),
        'azure': (255, 255, 240),
        'slate': (255, 0, 127),
        'chocolate': (30, 105, 210),
        'olive': (112, 255, 202),
        'orange': (0, 140, 255),
        'orchid': (255, 102, 224)
    }
    color_scalar = color_dict[color_string]
    return color_scalar


def draw_bbox(img, bbox, score, classes, track_id = -1, img_id = -1):
    if track_id == -1:
        color = (255*rand(), 255*rand(), 255*rand())
    else:
        color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate', 'olive', 'orange', 'orchid']
        color_name = color_list[track_id % 13]
        color = find_color_scalar(color_name)

    if img_id % 10 == 0:
        color = find_color_scalar('red')
    elif img_id != -1:
        color = find_color_scalar('blue')

    cv2.rectangle(img,
                  (bbox[0], bbox[1]),
                  (bbox[0]+ bbox[2], bbox[1] + bbox[3]),
                  color = color,
                  thickness = 3)

    cls_name = classes[0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if track_id == -1:
        cv2.putText(img,
                    #'{:s} {:.2f}'.format(cls_name, score),
                    '{:s}'.format(cls_name),
                    (bbox[0], bbox[1]-5),
                    font,
                    fontScale=0.8,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
    else:
        cv2.putText(img,
                    #'{:s} {:.2f}'.format("ID:"+str(track_id), score),
                    '{:s}'.format("ID:"+str(track_id)),
                    (bbox[0], bbox[1]-5),
                    font,
                    fontScale=0.8,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
    return img


def show_boxes_from_standard_json(json_file_path, classes, img_folder_path = None, output_folder_path = None, track_id = -1):
    dets = read_json_from_file(json_file_path)

    for det in dets:
        python_data = det

        if img_folder_path is None:
            img_path = os.path.join(python_data["image"]["folder"], python_data["image"]["name"])
        else:
            img_path = os.path.join(img_folder_path, python_data["image"]["name"])
        if is_image(img_path):    img = cv2.imread(img_path)

        candidates = python_data["candidates"]
        for candidate in candidates:
            bbox = np.array(candidate["det_bbox"]).astype(int)
            score = candidate["det_score"]
            if score >= bbox_thresh:
                img = draw_bbox(img, bbox, score, classes, track_id = track_id)

        if output_folder_path is not None:
            create_folder(output_folder_path)
            img_output_path = os.path.join(output_folder_path, python_data["image"]["name"])
            cv2.imwrite(img_output_path, img)
    return True
