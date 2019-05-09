'''
 test_poly_visualizer.py
 Unit Test for Visualizer of Polygon-level Segmentation
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on July 11th, 2018
'''
import sys, os

sys.path.append(os.path.abspath("../utility/"))
from utils_io_folder import create_folder, get_immediate_childfile_paths
from utils_io_file import is_image

sys.path.append(os.path.abspath("../standardize/convert/detect_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/keypoint_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/keypoint_track_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/poly_to_standard"))
from poly_visualizer import *

classes = ["person"]
def test_show_poly_from_standard_json():
    #json_file_path = "../standardize/convert/poly_to_standard/json/poly_results_standard_format.json"  # generated locally
    '''
    json_file_path = "/Users/guanghan.ning/Desktop/dev/polyrnn/poly.json"     # downloaded from server
    img_folder_path = "/Users/guanghan.ning/Desktop/working/douyin_4/images/"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/douyin_4/visualize/poly/"
    '''

    json_file_path = "/Users/guanghan.ning/Desktop/leftImg8bit_demoVideo/leftImg8bit/demoVideo/results/poly/cityscape.json"     # downloaded from server
    img_folder_path = "/Users/guanghan.ning/Desktop/leftImg8bit_demoVideo/leftImg8bit/demoVideo/images/"
    output_folder_path = "/Users/guanghan.ning/Desktop/leftImg8bit_demoVideo/leftImg8bit/demoVideo/visualize/poly/"

    create_folder(output_folder_path)
    show_poly_from_standard_json(json_file_path, classes, joint_pairs, joint_names, img_folder_path, output_folder_path, flag_track = True)
    print("Done!")


def test_make_video_from_images():
    '''
    img_folder_path = "/Users/guanghan.ning/Desktop/working/douyin_4/visualize/poly/"
    output_video_path = "/Users/guanghan.ning/Desktop/working/douyin_4/visualize/video_poly.mp4"
    '''
    img_folder_path = "/Users/guanghan.ning/Desktop/leftImg8bit_demoVideo/leftImg8bit/demoVideo/visualize/poly/"
    output_video_path = "/Users/guanghan.ning/Desktop/leftImg8bit_demoVideo/leftImg8bit/demoVideo/visualize/video_poly.mp4"

    img_paths = get_immediate_childfile_paths(img_folder_path)
    make_video_from_images(img_paths, output_video_path, fps=20, size=None, is_color=True, format="XVID")
    print("Done!")
    return


if __name__ == "__main__":
    test_show_poly_from_standard_json()
    test_make_video_from_images()
