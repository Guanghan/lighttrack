'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    November 7th, 2018

    Load keypoints from existing openSVAI data format
    and turn these keypoints into Graph structure for GCN

    Perform pose matching on these pairs. These pairs are from different persons.
    Unless their pose are very similar, normally their poses should not match.

'''
import numpy as np
import argparse
import torch

import sys, os
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../graph/utils/"))

from utils_json import *
from utils_io_folder import *

from keypoint_visualizer import *
from detection_visualizer import *

def test_visualization(dataset_str, dataset_split_str):
    if dataset_str == "posetrack_18":
        if dataset_split_str == "train":
            json_folder_path = "/export/guanghan/Data_2018_lighttrack/posetrack_data/gcn_openSVAI/train"
        elif dataset_split_str == "val":
            json_folder_path = "/export/guanghan/Data_2018_lighttrack/posetrack_data/gcn_openSVAI/val"
        elif dataset_split_str == "test":
            json_folder_path = "/export/guanghan/Data_2018_lighttrack/posetrack_data/gcn_openSVAI/val"

        json_file_paths = get_immediate_childfile_paths(json_folder_path)

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

                img, flag_match = visualize_graph_matching(candidate_dict_A, graph_A, candidate_dict_B, graph_B)
                match_str = "Match" if flag_match else "Not_Match"

                img_name = match_str + "_frame_" + str(candidate_dict_A["img_id"]) + "_" + str(track_id_A) + "_" + str(track_id_B) + ".jpg"
                img_path = os.path.join("/export/guanghan/temp2/", img_name)
                cv2.imwrite(img_path, img)
    return


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
        x = keypoints[3*id] - x0
        y = keypoints[3*id+1] - y0

        score = keypoints[3*id+2]
        graph[id] = (int(x), int(y))
    return graph, flag_pass_check


#----------------------------------------------------
from utils.io import IO
from utils.gcn_model import Model
from utils.processor_siamese_gcn import SGCN_Processor
import torchlight

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
        parser.set_defaults(config='config/inference.yaml')
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
        print(distance)
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

    if img_path_A == img_path_B:
        img = cv2.imread(img_path_A)
        print(img.shape)

        # draw person bbox and keypoints on img A
        pose_keypoints_2d = candidate_A["keypoints"]
        joints = reshape_keypoints_into_joints(pose_keypoints_2d)
        img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

        # draw person bbox and keypoints on img B
        pose_keypoints_2d = candidate_B["keypoints"]
        joints = reshape_keypoints_into_joints(pose_keypoints_2d)
        img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

        # draw match score on img A
        bbox = candidate_A["bbox"]
        track_id_A = candidate_A["track_id"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = find_color_scalar('red')
        cv2.putText(img,
                    'ID:{}, {}, dist:{:.2f}'.format(track_id_A, match_str, dist),
                    (int(bbox[0]), int(bbox[1])),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
        color = find_color_scalar('blue')
        cv2.putText(img,
                    'Frame #: {}'.format(candidate_A["img_id"]),
                    (30, 30),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)

        # draw match or not on img B
        bbox = candidate_B["bbox"]
        track_id_B = candidate_B["track_id"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = find_color_scalar('red')
        cv2.putText(img,
                    'ID:{}, {}, dist:{:.2f}'.format(track_id_B, match_str, dist),
                    (int(bbox[0]), int(bbox[1])),
                    font,
                    fontScale=1,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
        color = find_color_scalar('blue')
    return img, flag_match


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
