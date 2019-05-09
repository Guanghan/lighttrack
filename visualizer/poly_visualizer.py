'''
 poly_visualizer.py
 Visualizer for Detection, Human Pose Estimation, Segmentation, etc
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on July 7th, 2018
'''
import sys, os
sys.path.insert(0, os.path.abspath("../detect_to_standard/"))
from detection_visualizer import *

sys.path.append(os.path.abspath("../keypoint_to_standard/"))
from keypoint_visualizer import *

import json
import numpy as np
import cv2

draw_threshold = 0.2

def show_poly_from_standard_json(json_file_path, classes, joint_pairs, joint_names, img_folder_path = None, output_folder_path = None, flag_track= False):
    # Visualizing: Detection + Pose Estimation
    dets = read_json_from_file(json_file_path)

    for det in dets:
        python_data = det

        if img_folder_path is None:
            img_path = os.path.join(python_data["image"]["folder"], python_data["image"]["name"])
        else:
            img_path = os.path.join(img_folder_path, python_data["image"]["name"])
        print(img_path)
        if is_image(img_path):    img = cv2.imread(img_path)

        candidates = python_data["candidates"]
        for candidate in candidates:
            bbox = np.array(candidate["det_bbox"]).astype(int)
            score = candidate["det_score"]

            if score < draw_threshold: continue

            if flag_track is True:
                track_id = candidate["track_id"]
                if track_id == 55: continue
                if track_id == 49: continue
                if track_id == 120: continue
                img = draw_bbox(img, bbox, score, classes, track_id = track_id)
            else:
                img = draw_bbox(img, bbox, score, classes)

            # draw polys on the image
            polys = candidate["segmentation"]
            img_copy = img.copy()
            if flag_track is True:
                track_id = candidate["track_id"]
                if track_id == 55: continue
                if track_id == 49: continue
                if track_id == 120: continue
                img = draw_poly(img, polys, track_id = track_id)
            else:
                img = draw_poly(img, polys)
            img = img*0.3 + img_copy*0.7

            pose_keypoints_2d = candidate["pose_keypoints_2d"]
            joints = reshape_keypoints_into_joints(pose_keypoints_2d)

            if flag_track is True:
                #track_id = candidate["track_id"]
                #img = show_poses_from_python_data(img, joints, joint_pairs, joint_names, track_id = track_id)
                if track_id == 55: continue
                if track_id == 49: continue
                if track_id == 120: continue
                img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)
            else:
                img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

            if output_folder_path is not None:
                create_folder(output_folder_path)
                img_output_path = os.path.join(output_folder_path, python_data["image"]["name"])
                cv2.imwrite(img_output_path, img)
    return


def make_video_from_images(img_paths, outvid_path, fps=25, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for img_path in img_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        img = imread(img_path)
        if img is None:
            print(img_path)
            continue
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid_path, fourcc, float(fps), size, is_color)

        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def make_gif_from_images(img_paths, outgif_path):
    import imageio
    with imageio.get_writer(outgif_path, mode='I') as writer:
        for img_path in img_paths:
            image = imageio.imread(img_path)
            # Do sth to make gif file smaller
            # 1) change resolution
            # 2) change framerate
            writer.append_data(image)
    print("Gif made!")
    return


def draw_poly(mask, poly, track_id=-1):
    """
    Draw a polygon in the img.

    Args:
    img: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if track_id == -1:
        color = (255*rand(), 255*rand(), 255*rand())
    else:
        color_list = ['purple', 'yellow', 'blue', 'green', 'red', 'skyblue', 'navyblue', 'azure', 'slate', 'chocolate', 'olive', 'orange', 'orchid']
        color_name = color_list[track_id % 13]
        color = find_color_scalar(color_name)

    poly = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [poly], color=color)
    return mask
