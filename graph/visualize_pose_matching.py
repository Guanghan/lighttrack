'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    November 5th, 2018

    Load keypoints from existing openSVAI data format
    and turn these keypoints into Graph structure for GCN

    Perform pose matching on these pairs.
    Output the image indicating whther they match or not.
'''
import numpy as np
import argparse
import torch

import sys, os
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath("visualizer"))
sys.path.append(os.path.abspath("graph"))

from utils_json import *
from utils_io_folder import *

from keypoint_visualizer import *
from detection_visualizer import *

def test_visualization(dataset_str, dataset_split_str):
    if dataset_str == "posetrack_18":
        if dataset_split_str == "train":
            json_folder_path = "data/Data_2018/posetrack_data/gcn_openSVAI/train"
        elif dataset_split_str == "val":
            json_folder_path = "data/Data_2018/posetrack_data/gcn_openSVAI/val"
        elif dataset_split_str == "test":
            json_folder_path = "data/Data_2018/posetrack_data/gcn_openSVAI/val"

        json_file_paths = get_immediate_childfile_paths(json_folder_path)

        graph_pair_list_all = []
        for json_file_path in json_file_paths:
            visualize_graph_pairs_from_json(json_file_path)
    return


def visualize_graph_pairs_from_json(json_file_path):
    python_data = read_json_from_file(json_file_path)
    num_imgs = len(python_data)

    track_id_dict = {}
    for track_id in range(100):
        track_id_dict[track_id] = []

    max_track_id = -1
    for img_id in range(num_imgs):
        image_id = python_data[img_id]["image"]["id"]
        candidates = python_data[img_id]["candidates"]
        image_path = os.path.join(python_data[img_id]["image"]["folder"],
                                  python_data[img_id]["image"]["name"])

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
                              "img_path": image_path,
                              "bbox": bbox,
                              "keypoints":keypoints}
            track_id_dict[track_id].append(candidate_dict)

    graph_pair_list_all = []
    for track_id in range(max_track_id):
        candidate_dict_list = track_id_dict[track_id]
        candidate_dict_list_sorted = sorted(candidate_dict_list, key=lambda k:k['img_id'])

        visualize_graph_pairs(candidate_dict_list_sorted, track_id)
    return


def visualize_graph_pairs(candidate_dict_list_sorted, track_id):
    num_dicts = len(candidate_dict_list_sorted)
    graph_pair_list = []
    #for dict_id in range(num_dicts - 1):
    for dict_id in range(num_dicts - 5):
        candidate_dict_curr = candidate_dict_list_sorted[dict_id]
        #candidate_dict_next = candidate_dict_list_sorted[dict_id + 1]
        candidate_dict_next = candidate_dict_list_sorted[dict_id + 5]

        if candidate_dict_next["img_id"] - candidate_dict_curr["img_id"] >= 10:
            continue
        if candidate_dict_next["img_id"] - candidate_dict_curr["img_id"] <= 4:
            continue
        #print("current_dict_imgid: {}, next_dict_imgid: {}".format(candidate_dict_curr["img_id"], candidate_dict_next["img_id"]))

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

        concat_img, flag_match = visualize_graph_matching(candidate_dict_curr, graph_curr, candidate_dict_next, graph_next)
        match_str = "Match" if flag_match else "Not_Match"

        img_name = match_str + "_" + str(candidate_dict_curr["img_id"]) + "_" + str(candidate_dict_next["img_id"]) + "_" + str(track_id) + ".jpg"
        img_path = os.path.join("/export/guanghan/temp/", img_name)
        cv2.imwrite(img_path, concat_img)
    return


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
        x = keypoints[3*id] - x0
        y = keypoints[3*id+1] - y0

        score = keypoints[3*id+2]
        graph[id] = (int(x), int(y))
    return graph, flag_pass_check

#----------------------------------------------------
from gcn_utils.io import IO
from gcn_utils.gcn_model import Model
from gcn_utils.processor_siamese_gcn import SGCN_Processor
import torchlight

#class Pose_Matcher(IO):
class Pose_Matcher(SGCN_Processor):
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        return


    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=False,
            parents=[parent_parser],
            description='Graph Convolution Network for Pose Matching')
        #parser.set_defaults(config='config/inference.yaml')
        parser.set_defaults(config='graph/config/inference.yaml')
        return parser


    def inference(self, data_1, data_2):
        self.model.eval()

        with torch.no_grad():
            data_1 = torch.from_numpy(data_1)
            data_1 = data_1.unsqueeze(0)
            data_1 = data_1.float().to(self.dev)

            data_2 = torch.from_numpy(data_2)
            data_2 = data_2.unsqueeze(0)
            data_2 = data_2.float().to(self.dev)

            feature_1, feature_2 = self.model.forward(data_1, data_2)

        # euclidian distance
        diff = feature_1 - feature_2
        dist_sq = torch.sum(pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        margin = 0.2
        distance = dist.data.cpu().numpy()[0]
        print("_____ Pose Matching: [dist: {:04.2f}]". format(distance))
        if dist >= margin:
            return False, distance  # Do not match
        else:
            return True, distance # Match


def visualize_graph_matching(candidate_A, graph_A, candidate_B, graph_B):
    img_path_root = "/export/guanghan/Data_2018/posetrack_data/"
    img_path_A = os.path.join(img_path_root, candidate_A["img_path"])
    img_path_B = os.path.join(img_path_root, candidate_B["img_path"])

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    flag_match, dist = pose_matching(data_A, data_B)
    match_str = "Match" if flag_match else "Not_Match"

    if img_path_A != img_path_B:
        img_A = cv2.imread(img_path_A)
        img_B = cv2.imread(img_path_B)
        #print(img_A.shape)

        # draw person bbox and keypoints on img A
        pose_keypoints_2d = candidate_A["keypoints"]
        joints = reshape_keypoints_into_joints(pose_keypoints_2d)
        img_A = show_poses_from_python_data(img_A, joints, joint_pairs, joint_names)

        # draw person bbox and keypoints on img B
        pose_keypoints_2d = candidate_B["keypoints"]
        joints = reshape_keypoints_into_joints(pose_keypoints_2d)
        img_B = show_poses_from_python_data(img_B, joints, joint_pairs, joint_names)

        # draw match score on img A
        bbox = candidate_A["bbox"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = find_color_scalar('red')
        cv2.putText(img_A,
                    '{}, dist:{:.2f}'.format(match_str, dist),
                    (int(bbox[0]), int(bbox[1]-5)),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
        color = find_color_scalar('blue')
        cv2.putText(img_A,
                    'Frame #: {}'.format(candidate_A["img_id"]),
                    (30, 30),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)

        # draw match or not on img B
        bbox = candidate_B["bbox"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = find_color_scalar('red')
        cv2.putText(img_B,
                    '{}, dist:{:.2f}'.format(match_str, dist),
                    (int(bbox[0]), int(bbox[1]-5)),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
        color = find_color_scalar('blue')
        cv2.putText(img_B,
                    'Frame #: {}'.format(candidate_B["img_id"]),
                    (30, 30),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)

        # concat the two images
        img_concat = cv2.hconcat([img_A, img_B])

    return img_concat, flag_match


def graph_pair_to_data(sample_graph_pair):
    data_numpy_pair = []
    for siamese_id in range(2):
        # fill data_numpy
        data_numpy = np.zeros((2, 1, 15, 1))

        pose = sample_graph_pair[:][siamese_id]
        data_numpy[0, 0, :, 0] = [x[0] for x in pose]
        data_numpy[1, 0, :, 0] = [x[1] for x in pose]
        data_numpy_pair.append(data_numpy)
    return data_numpy_pair[0], data_numpy_pair[1]


global pose_matcher
pose_matcher = Pose_Matcher()
def pose_matching(graph_A_data, graph_B_data):
    flag_match, dist = pose_matcher.inference(graph_A_data, graph_B_data)
    return flag_match, dist


if __name__ == "__main__":
    test_visualization("posetrack_18", "val")
