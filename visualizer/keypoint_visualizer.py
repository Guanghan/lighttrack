'''
 keypoint_visualizer.py
 Visualizer for Human Pose Estimation
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on June 18th, 2018
'''
import cv2
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../../../utility/"))
from utils_io_file import is_image
from utils_io_folder import create_folder
from utils_json import read_json_from_file

flag_color_sticks = True
flag_only_draw_sure = False
#keypoints_mode = "COCO"
keypoints_mode = "PoseTrack"

if keypoints_mode == "COCO":
    print("COCO order.")
    joint_names = ['nose',  'left eye','right eye','left ear','right ear','left shoulder','right shoulder',
                   'left elbow','right elbow','left wrist','right wrist','left pelvis','right pelvis',
                   'left knee','right knee','left ankle','right ankle'] # COCO order
                   #{0-nose    1-Leye    2-Reye    3-Lear    4Rear    5-Lsho    6-Rsho    7-Lelb    8-Relb    9-Lwri    10-Rwri    11-Lhip    12-Rhip    13-Lkne    14-Rkne    15-Lank    16-Rank}　
    joint_pairs = [['nose', 'right shoulder', 'yellow'],
                   ['nose', 'left shoulder', 'yellow'],
                   ['right shoulder', 'right elbow', 'blue'],
                   ['right elbow', 'right wrist', 'green'],
                   ['left shoulder', 'left elbow', 'blue'],
                   ['left elbow', 'left wrist', 'green'],
                   ['right shoulder', 'right pelvis', 'yellow'],
                   ['left shoulder', 'left pelvis', 'yellow'],
                   ['right pelvis', 'right knee', 'red'],
                   ['right knee', 'right ankle', 'skyblue'],
                   ['left pelvis', 'left knee', 'red'],
                   ['left knee', 'left ankle', 'skyblue'],

                   ['left eye', 'left ear', 'skyblue'],
                   ['left eye', 'nose', 'skyblue'],
                   ['right eye', 'right ear', 'skyblue'],
                   ['right eye', 'nose', 'skyblue']]
elif keypoints_mode == "MPII":
    print("MPII order.")
    joint_names = ['head', 'upper neck', 'right shoulder', 'right elbow', 'right wrist',
               'left shoulder', 'left elbow', 'left wrist', 'right pelvis',
               'right knee', 'right ankle', 'left pelvis', 'left knee', 'left ankle',
               'background' ] # MPII or LSP order
    joint_pairs = [['head', 'upper neck', 'purple'],
                   ['upper neck', 'right shoulder', 'yellow'],
                   ['upper neck', 'left shoulder', 'yellow'],
                   ['right shoulder', 'right elbow', 'blue'],
                   ['right elbow', 'right wrist', 'green'],
                   ['left shoulder', 'left elbow', 'blue'],
                   ['left elbow', 'left wrist', 'green'],
                   ['right shoulder', 'right pelvis', 'yellow'],
                   ['left shoulder', 'left pelvis', 'yellow'],
                   ['right pelvis', 'right knee', 'red'],
                   ['right knee', 'right ankle', 'skyblue'],
                   ['left pelvis', 'left knee', 'red'],
                   ['left knee', 'left ankle', 'skyblue']]
elif keypoints_mode == "PoseTrack":
    print("PoseTrack order.")
    joint_names = ['right ankle', 'right knee', 'right pelvis', 'left pelvis',
                   'left knee', 'left ankle', 'right wrist',
                   'right elbow', 'right shoulder', 'left shoulder', 'left elbow', 'left wrist',
                   'upper neck', 'nose', 'head']
                    #{0-Rank    1-Rkne    2-Rhip    3-Lhip    4-Lkne    5-Lank    6-Rwri    7-Relb    8-Rsho    9-Lsho   10-Lelb    11-Lwri    12-neck  13-nose　14-TopHead}
    joint_pairs = [['head', 'upper neck', 'purple'],
                   ['upper neck', 'right shoulder', 'yellow'],
                   ['upper neck', 'left shoulder', 'yellow'],
                   ['right shoulder', 'right elbow', 'blue'],
                   ['right elbow', 'right wrist', 'green'],
                   ['left shoulder', 'left elbow', 'blue'],
                   ['left elbow', 'left wrist', 'green'],
                   ['right shoulder', 'right pelvis', 'yellow'],
                   ['left shoulder', 'left pelvis', 'yellow'],
                   ['right pelvis', 'right knee', 'red'],
                   ['right knee', 'right ankle', 'skyblue'],
                   ['left pelvis', 'left knee', 'red'],
                   ['left knee', 'left ankle', 'skyblue']]

color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate', 'olive', 'orange', 'orchid']


def show_poses_from_standard_json(json_file_path, joint_pairs, joint_names, img_folder_path = None, output_folder_path = None):
    poses = read_json_from_file(json_file_path)

    for pose in poses:
        python_data = pose

        if img_folder_path is None:
            img_path = os.path.join(python_data["image"]["folder"], python_data["image"]["name"])
        else:
            img_path = os.path.join(img_folder_path, python_data["image"]["name"])
        if is_image(img_path):    img = cv2.imread(img_path)

        candidates = python_data["candidates"]
        for candidate in candidates:
            pose_keypoints_2d = candidate["pose_keypoints_2d"]
            joints = reshape_keypoints_into_joints(pose_keypoints_2d)
            img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

        if output_folder_path is not None:
            create_folder(output_folder_path)
            img_output_path = os.path.join(output_folder_path, python_data["image"]["name"])
            cv2.imwrite(img_output_path, img)
    return


def show_poses_from_python_data(img, joints, joint_pairs, joint_names, flag_demo_poses = False, track_id = -1):
    img = add_joints_to_image(img, joints)

    if track_id == -1: # do pose estimation visualization
        img = add_joint_connections_to_image(img, joints, joint_pairs, joint_names)
    else:  # do pose tracking visualization
        candidate_joint_pairs = joint_pairs.copy()
        color_name = color_list[track_id % 6]
        for i in range(len(candidate_joint_pairs)):   candidate_joint_pairs[i][2] = color_name
        img = add_joint_connections_to_image(img, joints, candidate_joint_pairs, joint_names)

    if flag_demo_poses is True:
        cv2.imshow("pose image", img)
        cv2.waitKey(0.1)
    return img


def add_joints_to_image(img_demo, joints):
    for joint in joints:
        [i, j, sure] = joint
        #cv2.circle(img_demo, (i, j), radius=8, color=(255,255,255), thickness=2)
        cv2.circle(img_demo, (i, j), radius=2, color=(255,255,255), thickness=2)
    return img_demo


def add_joint_connections_to_image(img_demo, joints, joint_pairs, joint_names):
    for joint_pair in joint_pairs:
        ind_1 = joint_names.index(joint_pair[0])
        ind_2 = joint_names.index(joint_pair[1])
        if flag_color_sticks is True:
            color = find_color_scalar(joint_pair[2])
        else:
            color = find_color_scalar('red')

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1<= 5: continue
        if x2 <= 5 and y2<= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 > 0.5 and sure2 > 0.5:
            #cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=4)
    return img_demo


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


def reshape_keypoints_into_joints(pose_keypoints_2d):
    # reshape vector of length 3N into an array of shape [N, 3]
    num_keypoints = int(len(pose_keypoints_2d) / 3)
    joints = np.array(pose_keypoints_2d).reshape(num_keypoints, 3).astype(int)
    return joints
