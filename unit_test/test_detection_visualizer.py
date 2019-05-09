'''
 test_detection_visualizer.py
 Unit Test for Visualizer of object (human candidate) detection
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on June 18th, 2018
'''
import sys, os

sys.path.append(os.path.abspath("../utility/"))
from utils_io_folder import create_folder
from utils_io_file import is_image

sys.path.append(os.path.abspath("../standardize/convert/detect_to_standard"))
from detection_visualizer import *

def test_show_boxes_from_standard_json():
    json_file_path = "../standardize/convert/keypoint_to_standard/keypoint_CPN_to_standard.json"
    img_folder_path = "/Users/guanghan.ning/Desktop/working/hard_examples/"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/hard_examples_detection/"
    create_folder(output_folder_path)
    show_boxes_from_standard_json(json_file_path, classes, img_folder_path, output_folder_path)
    print("Done!")


if __name__ == "__main__":
    test_show_boxes_from_standard_json()
