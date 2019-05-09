'''
 test_keypoint_visualizer.py
 Unit Test for Visualizer of Human Pose Estimation
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on June 19th, 2018
'''
import sys, os

sys.path.append(os.path.abspath("../utility/"))
from utils_io_folder import create_folder, get_immediate_childfile_paths
from utils_io_file import is_image

sys.path.append(os.path.abspath("../standardize/convert/detect_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/keypoint_to_standard"))
sys.path.append(os.path.abspath("../standardize/convert/keypoint_track_to_standard"))
from visualizer import *

classes = ["person"]
def test_show_all_from_standard_json():
    #json_file_path = "../standardize/convert/keypoint_to_standard/json/keypoint_CPN_to_standard.json"
    #img_folder_path = "/Users/guanghan.ning/Desktop/working/hard_examples/"
    #output_folder_path = "/Users/guanghan.ning/Desktop/working/hard_examples_visualize/"
    json_file_path = "../standardize/convert/keypoint_to_standard/json/temp.json"
    img_folder_path = "/Users/guanghan.ning/Desktop/working/test_openSVAI/"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/test_openSVAI_visualize/"
    create_folder(output_folder_path)
    show_all_from_standard_json(json_file_path, classes, joint_pairs, joint_names, img_folder_path, output_folder_path)
    print("Done!")


def test_show_all_from_standard_json_track():
    json_file_path = "../standardize/convert/keypoint_track_to_standard/json/track_results_standard_format.json"
    img_folder_path = "/Users/guanghan.ning/Desktop/working/test_posetrack/"
    output_folder_path = "/Users/guanghan.ning/Desktop/working/test_posetrack_visualize/"
    create_folder(output_folder_path)
    show_all_from_standard_json(json_file_path, classes, joint_pairs, joint_names, img_folder_path, output_folder_path, flag_track = True)
    print("Done!")


def test_make_video_from_images():
    #img_folder_path = "/Users/guanghan.ning/Desktop/working/test_posetrack_visualize/"
    #output_video_path = "/Users/guanghan.ning/Desktop/working/video.mp4"

    #img_folder_path = "/Users/guanghan.ning/Desktop/working/visualize_douyin_3/track/"
    #output_video_path = "/Users/guanghan.ning/Desktop/working/visualize_douyin_3/video_fast.mp4"

    img_folder_path = "/Users/guanghan.ning/Desktop/working/exp4/pose/"
    output_video_path = "/Users/guanghan.ning/Desktop/working/exp4/video_fast.mp4"

    img_paths = get_immediate_childfile_paths(img_folder_path)
    make_video_from_images(img_paths, output_video_path, fps=20, size=None, is_color=True, format="XVID")
    print("Done!")
    return


def test_track_all(dataset = "test"):
    image_dir = "/export/guanghan/Data_2018_lighttrack/posetrack_data/"
    visualize_folder = "/export/guanghan/Data_2018_lighttrack/posetrack_results/lighttrack/visualize/"
    output_video_folder = "/export/guanghan/cpn/data"
    track_json_folder = "/export/guanghan/Data_2018/posetrack_results/{}/track/jsons_submission3".format(dataset)
    video_name_list = ["018630_mpii_test", "024156_mpii_test"]#["022699_mpii_test", "023730_mpii_test", "014307_mpii_test", "014313_mpii_test", "015868_mpii_test"]

    for video_name in video_name_list:
        json_name = os.path.basename(video_name) + ".json"
        track_json = os.path.join(track_json_folder, json_name)
        print("visualization!")
        create_folder(visualize_folder)
        if dataset == "val":
            image_subfolder = os.path.join(image_dir, "images/val", os.path.basename(video_name))
        elif dataset == "test":
            image_subfolder = os.path.join(image_dir, "images/test", os.path.basename(video_name))
        print(image_subfolder)

        visualize_subfolder = os.path.join(visualize_folder, os.path.basename(video_name))
        create_folder(visualize_subfolder)
        show_all_from_standard_json(track_json, classes, joint_pairs, joint_names, image_subfolder, visualize_subfolder, flag_track = True)

        img_paths = get_immediate_childfile_paths(visualize_subfolder)
        visualize_video_path = os.path.join(visualize_folder, os.path.basename(video_name)+".mp4")
        make_video_from_images(img_paths, visualize_video_path, fps=15, size=None, is_color=True, format="xvid")
    return


if __name__ == "__main__":
    #test_show_all_from_standard_json()
    #test_show_all_from_standard_json_track()
    #test_make_video_from_images()
    test_track_all()

