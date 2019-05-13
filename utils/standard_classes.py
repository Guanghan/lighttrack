'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    September 27th, 2018

    OpenSVAI standard format
'''
from utils_json import python_to_json, write_json_to_file

class StandardData():
    def __init__(self):
        self.python_data = {"version": "1.0"}

    def add_image_info(self, image_info):
        self.python_data["image"] = image_info

    def add_candidates(self, candidate_list):
        self.python_data["candidates"] = candidate_list

    def get_python_data(self):
        return self.python_data

    def get_json_str(self):
        json_str = python_to_json(self.python_data)
        return json_str

    def write_to_file(self, output_json_path):
        write_json_to_file(self.python_data, output_json_path)


class StandardImageInfo():
    """
    The image information class in SVAI data standard
    """
    def __init__(self):
        self.image_info = {}

    def add_image_folder(self, folder_path):
        self.image_info["folder"] = folder_path

    def add_image_name(self, image_name):
        self.image_info["name"] = image_name

    def add_image_id(self, image_id):
        self.image_info["id"] = image_id


class StandardCandidate():
    """
    The candidate class in SVAI data standard
    We can add a candidate to an existing python data easily
    """
    def __init__(self):
        self.candidate = {}

    def add_det_bbox(self, det_bbox):
        self.candidate["det_bbox"] = det_bbox

    def add_det_score(self, det_score):
        self.candidate["det_score"] = det_score

    def add_det_category(self, det_category):
        self.candidate["det_category"] = det_category

    def add_pose_order(self, pose_order):
        self.candidate["pose_order"] = pose_order

    def add_pose_keypoints_2d(self, pose_keypoints_2d):
        self.candidate["pose_keypoints_2d"] = pose_keypoints_2d

    def add_track_id(self, track_id):
        self.candidate["track_id"] = track_id

    def add_track_score(self, track_score):
        self.candidate["track_score"] = track_score
