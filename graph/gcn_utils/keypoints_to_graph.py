'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 22th, 2018

    Load keypoints from existing openSVAI data format
    and turn these keypoints into Graph structure for GCN

    N: # of batch_size
    M: # of instances within a frame (which is # of human candidates)
    V: # of graph nodes (which is 15)
'''
import numpy as np

import sys, os
sys.path.append(os.path.abspath("../../"))

from utils_json import *
from utils_io_folder import *

def load_data_for_gcn(dataset_str, dataset_split_str):
    if dataset_str == "posetrack_18":
        if dataset_split_str == "train":
            json_folder_path = "/export/guanghan/Data_2018/posetrack_data/gcn_openSVAI/train"
        elif dataset_split_str == "val":
            json_folder_path = "/export/guanghan/Data_2018/posetrack_data/gcn_openSVAI/val"
        elif dataset_split_str == "test":
            json_folder_path = "/export/guanghan/Data_2018/posetrack_data/gcn_openSVAI/val"

        json_file_paths = get_immediate_childfile_paths(json_folder_path)

        graph_pair_list_all = []
        for json_file_path in json_file_paths:
            graph_pair_list = load_graph_pairs_from_json(json_file_path)
            graph_pair_list_all.extend(graph_pair_list)

    return graph_pair_list_all


def load_graph_pairs_from_json(json_file_path):
    python_data = read_json_from_file(json_file_path)
    num_imgs = len(python_data)

    track_id_dict = {}
    for track_id in range(100):
        track_id_dict[track_id] = []

    max_track_id = -1
    for img_id in range(num_imgs):
        image_id = python_data[img_id]["image"]["id"]
        candidates = python_data[img_id]["candidates"]

        num_candidates = len(candidates)
        for candidate_id in range(num_candidates):
            candidate = candidates[candidate_id]
            track_id = candidate["track_id"]
            keypoints = candidate["pose_keypoints_2d"]
            bbox = candidate["det_bbox"]

            if track_id > max_track_id:
                max_track_id = track_id

            candidate_dict = {"track_id": track_id,
                              "img_id": image_id,
                              "bbox": bbox,
                              "keypoints":keypoints}
            track_id_dict[track_id].append(candidate_dict)

    graph_pair_list_all = []
    for track_id in range(max_track_id):
        candidate_dict_list = track_id_dict[track_id]
        candidate_dict_list_sorted = sorted(candidate_dict_list, key=lambda k:k['img_id'])

        graph_pair_list = get_graph_pairs(candidate_dict_list_sorted)
        graph_pair_list_all.extend(graph_pair_list)
    return  graph_pair_list_all


def get_graph_pairs(candidate_dict_list_sorted):
    num_dicts = len(candidate_dict_list_sorted)
    graph_pair_list = []
    for dict_id in range(num_dicts - 1):
        candidate_dict_curr = candidate_dict_list_sorted[dict_id]
        candidate_dict_next = candidate_dict_list_sorted[dict_id + 1]

        if candidate_dict_next["img_id"] - candidate_dict_curr["img_id"] >= 2:
            continue

        keypoints_curr = candidate_dict_curr["keypoints"]
        keypoints_next = candidate_dict_next["keypoints"]

        bbox_curr = candidate_dict_curr["bbox"]
        bbox_next = candidate_dict_next["bbox"]

        if validate_bbox(bbox_curr) is False: continue
        if validate_bbox(bbox_next) is False: continue

        graph_curr, flag_pass_check = keypoints_to_graph(keypoints_curr, bbox_curr)
        if flag_pass_check is False: continue

        graph_next, flag_pass_check = keypoints_to_graph(keypoints_next, bbox_next)
        if flag_pass_check is False: continue

        graph_pair = (graph_curr, graph_next)  # use tuple over list
        graph_pair_list.append(graph_pair)
    return graph_pair_list


def validate_bbox(bbox):
    x0, y0, w, h = bbox
    if w <= 100 or h <= 100:
        return False
    else:
        return True


def keypoints_to_graph(keypoints, bbox):
    num_elements = len(keypoints)
    num_keypoints = num_elements/3
    assert(num_keypoints == 15)

    x0, y0, w, h = bbox
    flag_pass_check = True

    graph = 15*[(0, 0)]
    for id in range(15):
        # normalize the corrdinates: mean 0, standard deviation 1
        x = keypoints[3*id] - x0
        y = keypoints[3*id+1] - y0

        score = keypoints[3*id+2]
        graph[id] = (int(x), int(y))
    return graph, flag_pass_check



python_data_example =  {
    "version": "1.0",
	"image": [
	       {
		      "folder": "images/bonn_5sec/000342_mpii",
		      "name": "00000001.jpg",
              "id" : 0,
	       }
    ],
    "candidates":[
        {
          "det_category" : 1,
          "det_bbox" : [300,300,100,100],
          "det_score" : [0.9],

          "pose_order" : [1,2,3],
          "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

		  "track_id": [0],
		  "track_score": [0.8],
        },
        {
          "det_category" : 2,
          "det_bbox" : [300,300,100,100],
          "det_score" : [0.1],

          "pose_order" : [1,2,3],
          "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

		  "track_id": [1],
		  "track_score": [0.6],
        }
     ]
}
