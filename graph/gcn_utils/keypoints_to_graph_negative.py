'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 29th, 2018

    Load keypoints from existing openSVAI data format
    and turn these keypoints into Graph structure for GCN

    Produce negative pairs.

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
            json_folder_path = "./data/Data_2018/posetrack_data/gcn_openSVAI/train"
        elif dataset_split_str == "val":
            json_folder_path = "./data/Data_2018/posetrack_data/gcn_openSVAI/val"
        elif dataset_split_str == "test":
            json_folder_path = "./data/Data_2018/posetrack_data/gcn_openSVAI/val"

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
    for img_id in range(num_imgs):
        for track_id_A in range(max_track_id):
            for track_id_B in range(max_track_id):
                if track_id_A == track_id_B: continue
                candidate_A_index_list = find(track_id_dict[track_id_A], "img_id", img_id)
                candidate_B_index_list = find(track_id_dict[track_id_B], "img_id", img_id)

                if candidate_A_index_list == []:
                    continue
                if candidate_B_index_list == []:
                    continue

                index_A = candidate_A_index_list[0]
                index_B = candidate_B_index_list[0]

                candidate_dict_A = track_id_dict[track_id_A][index_A]
                candidate_dict_B = track_id_dict[track_id_B][index_B]

                keypoints_A = candidate_dict_A["keypoints"]
                keypoints_B = candidate_dict_B["keypoints"]

                bbox_A = candidate_dict_A["bbox"]
                bbox_B = candidate_dict_B["bbox"]

                if validate_bbox(bbox_A) is False: continue
                if validate_bbox(bbox_B) is False: continue

                graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
                if flag_pass_check is False: continue

                graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
                if flag_pass_check is False: continue

                graph_pair = (graph_A, graph_B)  # use tuple over list
                graph_pair_list_all.append(graph_pair)

    return graph_pair_list_all


def find(lst, key, value):
    # find the index of a dict in list
    index_list = []
    for i, dic in enumerate(lst):
        if dic[key] == value:
            index_list.append(i)
    return index_list


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
