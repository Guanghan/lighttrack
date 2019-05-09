
'''
 test_temp.py
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on August 27th, 2018
'''
import sys, os

sys.path.append(os.path.abspath("../utility/"))
from utils_io_folder import create_folder
from utils_io_file import is_image

sys.path.append(os.path.abspath("../standardize/convert/detect_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/keypoint_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/keypoint_track_to_standard"))
from detection_visualizer import *
from keypoint_visualizer import *
from visualizer import *

def test_show_boxes_from_standard_json():
    json_file_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/results/pose/009602_mpii_test.json"
    img_folder_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/images/009602_mpii_test"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/visualize/detection"

    create_folder(output_folder_path)
    show_boxes_from_standard_json(json_file_path, classes, img_folder_path, output_folder_path)
    print("Done!")

def test_show_poses_from_standard_json():
    json_file_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/results/track/009602_mpii_test.json"
    img_folder_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/images/009602_mpii_test"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/visualize/pose/"
    create_folder(output_folder_path)
    #show_poses_from_standard_json(json_file_path, joint_pairs, joint_names, img_folder_path, output_folder_path)
    show_all_from_standard_json(json_file_path, classes, joint_pairs, joint_names, img_folder_path, output_folder_path, flag_track = False)
    print("Done!")

def test_show_all_from_standard_json_track():
    json_file_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/results/track/009602_mpii_test.json"
    img_folder_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/images/009602_mpii_test"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/POSETRACK_visualize/visualize/track/"
    create_folder(output_folder_path)
    show_all_from_standard_json(json_file_path, classes, joint_pairs, joint_names, img_folder_path, output_folder_path, flag_track = True)
    print("Done!")

if __name__ == "__main__":
    test_show_boxes_from_standard_json()
    test_show_poses_from_standard_json()
    test_show_all_from_standard_json_track()
