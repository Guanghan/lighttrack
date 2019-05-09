'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    August 7th, 2018
    Calculate the BGR mean values for PoseTrack 2018 dataset
'''
from utils_io_folder import *
from utils_io_file import *
from utils_json import *
import cv2
import numpy as np

width = 288
ht = 384


def get_mean_from_whole_img():
    dataset_folder = "../data/Data_2018/posetrack_data/images/val"
    subfolder_paths = get_immediate_subfolder_paths(dataset_folder)

    B_sum = G_sum = R_sum = 0
    img_ct = 0
    for subfolder_path in subfolder_paths:
        img_paths = get_immediate_childfile_paths(subfolder_path, ext= "jpg")
        img_ct += len(img_paths)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (width, ht))
            B, G, R = cal_BGR_for_img(img_resized)
            B_sum += B
            G_sum += G
            R_sum += R
    B_mean = B_sum * 1.0 / img_ct
    G_mean = G_sum * 1.0 / img_ct
    R_mean = R_sum * 1.0 / img_ct
    print("BGR mean values: {:.3f}{:.3f}{:.3f}".format(B_mean, G_mean, R_mean))


def get_mean_from_human_patches():
    gt_path = "posetrack_merged_train_18.json"
    #gt_path = "posetrack_merged_val_18.json"
    gt_python_data = read_json_from_file(gt_path)
    anns = gt_python_data["annotations"]
    images_info = gt_python_data["images"]

    B_sum = G_sum = R_sum = 0
    img_ct = len(anns)
    for candidate_id, ann in enumerate(anns):
        if "keypoints" not in ann:
            img_ct -= 1
            continue

        # (a) create a bbox based on the keypoints
        bbox = get_bbox_from_keypoints(ann["keypoints"])
        bbox = x1y1x2y2_to_xywh(bbox)
        # (b) or use the bbox given by the annotations
        if "bbox" in ann and bbox == [0, 0, 2, 2]:
            bbox = ann["bbox"]
        if bbox == [0, 0, 2, 2]:
            img_ct -= 1
            continue

        # (2) imgpath
        image_id = ann["image_id"]
        index_list = find(images_info, key="frame_id", value=image_id)
        assert(len(index_list) > 0)
        imgname = images_info[index_list[0]]["file_name"]
        prefix = '../data/Data_2018/posetrack_data/'
        imgpath = os.path.join(prefix, imgname)

        img = cv2.imread(imgpath)
        x1 = int(bbox[0])
        x2 = int(x1 + bbox[2])
        y1 = int(bbox[1])
        y2 = int(y1 + bbox[3])
        if x1< 0: x1=0
        if y1< 0: y1=0
        if x2 <= 0 or y2 <= 0 or x2 <= x1 or y2 <= y1:
            img_ct -= 1
            continue

        img_roi = img[y1:y2, x1:x2, :]
        if img_roi.shape[0] == 0 or img_roi.shape[1] == 0:
            img_ct -= 1
            continue
        img_resized = cv2.resize(img_roi, (width, ht))

        B, G, R = cal_BGR_for_img(img_resized)
        B_sum += B
        G_sum += G
        R_sum += R
        print("BGR sum values: {:.3f}, {:.3f}, {:.3f}".format(B_sum, G_sum, R_sum))

    B_mean = B_sum * 1.0 / img_ct
    G_mean = G_sum * 1.0 / img_ct
    R_mean = R_sum * 1.0 / img_ct
    print("BGR mean values: {:.3f}, {:.3f}, {:.3f}".format(B_mean, G_mean, R_mean))


def cal_BGR_for_img(img):
    B_img = img[:, :, 0]
    G_img = img[:, :, 1]
    R_img = img[:, :, 2]

    B_mean = np.mean(B_img)
    G_mean = np.mean(G_img)
    R_mean = np.mean(R_img)
    return B_mean, G_mean, R_mean


def get_bbox_from_keypoints(keypoints_python_data):
    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(int(num_keypoints / 3)):
        x = keypoints_python_data[3 * keypoint_id]
        y = keypoints_python_data[3 * keypoint_id + 1]
        vis = keypoints_python_data[3 * keypoint_id + 2]
        if vis != 0:
            x_list.append(x)
            y_list.append(y)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    scale = 0.2  # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    return bbox


def enlarge_bbox(bbox, scale):
    assert(scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def find(lst, key, value):
    # find the index of a dict in list
    index_list = []
    for i, dic in enumerate(lst):
        if dic[key] == value:
            index_list.append(i)
    return index_list


if __name__ == "__main__":
    get_mean_from_human_patches()
