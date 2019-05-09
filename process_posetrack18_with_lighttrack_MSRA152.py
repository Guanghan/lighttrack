'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 3rd, 2018
    LightTrack: A Generic Framework for Online Top-Down Human Pose Tracking
'''
import argparse

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf

# import Network
from network_MSRA152 import Network

# pose estimation utils
from HPE.dataset import Preprocessing
from HPE.config import cfg
from tfflat.base import Tester
from tfflat.utils import mem_info
from tfflat.logger import colorlogger
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

# import GCN utils
from graph import visualize_pose_matching
from graph  .visualize_pose_matching import *

# import my own utils
import sys, os, time
sys.path.append(os.path.abspath("./graph/"))
from utils_json import *
from visualizer import *
from utils_io_folder import *

flag_visualize = True
flag_nms = False #Default is False, unless you know what you are doing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, dest='test_model', default="weights/MSRA152/snapshot_285.ckpt")
    parser.add_argument('--dataset_split', '-s', type=str, dest='dataset_split', default="posetrack18_val")
    parser.add_argument('--det_or_gt', '-e', type=str, dest='det_or_gt', default="det")
    args = parser.parse_args()

    args.bbox_thresh = 0.4
    args.pose_matching_threshold = 1.0

    assert args.test_model, 'no model is provided.'
    return args


def initialize_parameters():
    global nms_method, nms_thresh, min_scores, min_box_size
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.

    global keyframe_interval, enlarge_scale
    keyframe_interval = 2 # choice examples: [2, 3, 5, 8]
    enlarge_scale = 0.2

    global video_name, img_id
    return


def light_track(pose_estimator,
                annotation_json_file_path, output_json_path,
                image_folder, visualize_folder, output_video_path):
    precomputed_dets = load_gt_dets_mot(annotation_json_file_path)  # mode 2
    num_imgs = len(precomputed_dets)

    # process the frames sequentially
    keypoints_list = []
    bbox_dets_list = []
    frame_prev = -1
    frame_cur = 0
    next_id = 0
    bbox_dets_list_list = []
    keypoints_list_list = []

    flag_mandatory_keyframe = False
    img_id = -1
    while img_id < num_imgs-1:
        img_id += 1
        gt_data = precomputed_dets[img_id]

        if gt_data == []:
            # no gt annotation OR detection is available, so keep tracking
            flag_keep_tracking = True
            image_id = img_id

            if img_id == 0:
                img_path = os.path.join(image_folder, "000000.jpg")
            else:
                img_path = next_img_path(img_path)
        else:
            # load key-frame information
            flag_keep_tracking = False
            #image_id = gt_data[0]["image_id"] - 1  # start from 1 if using GT
            image_id = gt_data[0]["image_id"]  # start from 0 if using DET
            print("Current tracking: [image_id:{}]".format(image_id))
            assert(image_id == img_id)

            img_path = gt_data[0]["imgpath"]

        frame_cur = img_id
        if (frame_cur == frame_prev):
            frame_prev -= 1

        ''' KEYFRAME: loading results from other modules '''
        # if no gt annotation is available
        if flag_keep_tracking:
            bbox_dets_list = []  # keyframe: start from empty
            keypoints_list = []  # keyframe: start from empty

            # add empty result
            bbox_det_dict = {"img_id":img_id,
                                  "det_id":  0,
                                  "track_id": -1,
                                  "imgpath": img_path,
                                  "bbox": [0, 0, 2, 2]}
            bbox_dets_list.append(bbox_det_dict)

            keypoints_dict = {"img_id":img_id,
                                   "det_id": 0,
                                   "track_id": -1,
                                   "imgpath": img_path,
                                   "keypoints": []}
            keypoints_list.append(keypoints_dict)

            bbox_dets_list_list.append(bbox_dets_list)
            keypoints_list_list.append(keypoints_list)

        elif is_keyframe(img_id, keyframe_interval) or flag_mandatory_keyframe:
            flag_mandatory_keyframe = False

            bbox_dets_list = []  # keyframe: start from empty
            keypoints_list = []  # keyframe: start from empty

            num_dets = len(precomputed_dets[img_id])
            print("Keyframe: {} detections".format(num_dets))
            for det_id in range(num_dets):
                # obtain bbox position and track id
                bbox_gt = get_bbox_from_gt(precomputed_dets, img_id, det_id)

                # enlarge bbox by 20% with same center position
                bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_gt)
                bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
                bbox_gt = x1y1x2y2_to_xywh(bbox_in_xywh)

                # Keyframe: use provided bbox
                bbox_det = bbox_gt
                if bbox_det[2] <= 0 or bbox_det[3] <= 0 or bbox_det[2] > 2000 or bbox_det[3] > 2000:
                    bbox_det = [0, 0, 2, 2]
                    continue

                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":det_id,
                                 "imgpath": img_path,
                                 "bbox":bbox_det}
                # obtain keypoints for each bbox position in the keyframe
                keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]
                keypoints_gt = keypoints

                if img_id == 0:
                    track_id = next_id
                    next_id += 1
                else:
                    track_id = get_track_id_SpatialConsistency(bbox_gt, bbox_dets_list_list, img_id)
                    if track_id == -1:
                        track_id = get_track_id_SGCN(bbox_gt, bbox_dets_list_list, keypoints_gt, keypoints_list_list, img_id)

                    if track_id == -1 and not bbox_invalid(bbox_det):
                        track_id = next_id
                        next_id += 1

                if bbox_invalid(bbox_det):
                    track_id = -1
                    keypoints = []

                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":det_id,
                                 "track_id":track_id,
                                 "imgpath": img_path,
                                 "bbox":bbox_det}
                bbox_dets_list.append(bbox_det_dict)

                # update current frame keypoints
                keypoints_dict = {"img_id":img_id,
                                  "det_id":det_id,
                                  "track_id":track_id,
                                  "imgpath": img_path,
                                  "keypoints":keypoints}
                keypoints_list.append(keypoints_dict)

            # update frame
            bbox_dets_list_list.append(bbox_dets_list)
            keypoints_list_list.append(keypoints_list)
            frame_prev = frame_cur

        else:
            ''' NOT KEYFRAME: multi-target pose tracking '''
            bbox_dets_list_next = []
            keypoints_list_next = []

            num_dets = len(keypoints_list)

            if num_dets == 0:
                flag_mandatory_keyframe = True

            for det_id in range(num_dets):
                keypoints = keypoints_list[det_id]["keypoints"]

                # for non-keyframes, the tracked target preserves its track_id
                track_id = keypoints_list[det_id]["track_id"]

                # next frame bbox
                bbox_det_next = get_bbox_from_keypoints(keypoints)
                if bbox_det_next[2] == 0 or bbox_det_next[3] == 0:
                    bbox_det_next = [0, 0, 2, 2]
                assert(bbox_det_next[2] != 0 and bbox_det_next[3] != 0) # width and height must not be zero
                bbox_det_dict_next = {"img_id":img_id,
                                     "det_id":det_id,
                                     "track_id":track_id,
                                     "imgpath": img_path,
                                     "bbox":bbox_det_next}

                # next frame keypoints
                keypoints_next = inference_keypoints(pose_estimator, bbox_det_dict_next)[0]["keypoints"]

                # check whether the target is lost
                target_lost = is_target_lost(keypoints_next)

                if target_lost is False:
                    bbox_dets_list_next.append(bbox_det_dict_next)
                    keypoints_dict_next = {"img_id":img_id,
                                           "det_id":det_id,
                                           "track_id":track_id,
                                           "imgpath": img_path,
                                           "keypoints":keypoints_next}
                    keypoints_list_next.append(keypoints_dict_next)

                else:
                    # remove this bbox, do not register its keypoints
                    bbox_det_dict_next = {"img_id":img_id,
                                          "det_id":  det_id,
                                          "track_id": -1,
                                          "imgpath": img_path,
                                          "bbox": [0, 0, 2, 2]}
                    bbox_dets_list_next.append(bbox_det_dict_next)

                    keypoints_null = 45*[0]
                    keypoints_dict_next = {"img_id":img_id,
                                           "det_id":det_id,
                                           "track_id":track_id,
                                           "imgpath": img_path,
                                           "keypoints": []}
                    keypoints_list_next.append(keypoints_dict_next)
                    print("Target lost. Process this frame again as keyframe. \n\n\n")
                    flag_mandatory_keyframe = True

                    if img_id not in [0]:
                        img_id -= 1
                    break

            # update frame
            if flag_mandatory_keyframe is False:
                bbox_dets_list = bbox_dets_list_next
                keypoints_list = keypoints_list_next
                bbox_dets_list_list.append(bbox_dets_list)
                keypoints_list_list.append(keypoints_list)
                frame_prev = frame_cur

    # convert results into openSVAI format
    print("Export results in openSVAI standard format...")
    poses_standard = pose_to_standard_mot(keypoints_list_list, bbox_dets_list_list)
    #json_str = python_to_json(poses_standard)
    #print(json_str)

    # output json file
    pose_json_folder, _ = get_parent_folder_from_path(output_json_path)
    create_folder(pose_json_folder)
    write_json_to_file(poses_standard, output_json_path)

    # visualization
    if flag_visualize is True:
        create_folder(visualize_folder)
        show_all_from_standard_json(output_json_path, classes, joint_pairs, joint_names, image_folder, visualize_folder, flag_track = True)
        print("Pose Estimation Finished!")

        img_paths = get_immediate_childfile_paths(visualize_folder)
        make_video_from_images(img_paths, output_video_path, fps=10, size=None, is_color=True, format="XVID")


def get_track_id_SGCN(bbox_gt, bbox_dets_list_list, keypoints_gt, keypoints_list_list, img_id):
    assert(len(bbox_dets_list_list) == len(keypoints_list_list))

    # get bboxes from previous frame
    bbox_dets_list = bbox_dets_list_list[img_id - 1]
    keypoints_list = keypoints_list_list[img_id - 1]

    for det_id, bbox_det_dict in enumerate(bbox_dets_list):
        bbox_det = bbox_det_dict["bbox"]
        # check the pose matching score
        keypoints_dict = keypoints_list[det_id]
        keypoints = keypoints_dict["keypoints"]

        pose_matching_score = get_pose_matching_score(keypoints_gt, keypoints, bbox_gt, bbox_det)

        if pose_matching_score <= args.pose_matching_threshold:
            # match the target based on the pose matching score
            track_id = bbox_det_dict["track_id"]
            return track_id

    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1
    return track_id


def get_track_id_SpatialConsistency(bbox_gt, bbox_dets_list_list, img_id):
    # get bboxes from previous frame
    bbox_dets_list = bbox_dets_list_list[img_id - 1]

    thresh = 0.3
    max_iou_score = -1000
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_dets_list):
        bbox_det = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_gt)
        boxB = xywh_to_x1y1x2y2(bbox_det)
        iou_score = iou(boxA, boxB)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        return bbox_dets_list[max_index]["track_id"]
    else:
        return -1


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    if keypoints_A == [] or keypoints_B == []:
        print("graph not correctly generated!")
        return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated!")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    start = time.time()
    flag_match, dist = pose_matching(data_A, data_B)
    end = time.time()
    return dist


def is_target_lost(keypoints, method="max_average"):
    num_keypoints = int(len(keypoints) / 3.0)
    if method == "average":
        # pure average
        score = 0
        for i in range(num_keypoints):
            score += keypoints[3*i + 2]
        score /= num_keypoints*1.0
        print("target_score: {}".format(score))
    elif method == "max_average":
        score_list = keypoints[2::3]
        score_list_sorted = sorted(score_list)
        top_N = 4
        assert(top_N < num_keypoints)
        top_scores = [score_list_sorted[-i] for i in range(1, top_N+1)]
        score = sum(top_scores)/top_N
    if score < 0.6:
        return True
    else:
        return False


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def load_gt_dets_mot(json_folder_input_path):
    ''' load all detections in a video by reading json folder'''
    if json_folder_input_path.endswith(".json"):
        json_file_path = json_folder_input_path
        dets_standard = read_json_from_file(json_file_path)
    else:
        dets_standard = batch_read_json(json_folder_input_path)

    print("Using detection threshold: ", args.bbox_thresh)
    dets = standard_to_dicts(dets_standard, bbox_thresh = args.bbox_thresh)

    print("Number of imgs: {}".format(len(dets)))
    return dets


def batch_read_json(json_folder_path):
    json_paths = get_immediate_childfile_paths(json_folder_path, ext=".json")

    dets = []
    for json_path in json_paths:
        python_data = read_json_from_file(json_path)
        dets.append(python_data)
    return dets


def standard_to_dicts(dets_standard, bbox_thresh = 0):
    # standard detection format to CPN detection format
    num_dets = len(dets_standard)
    dets_CPN_list = []
    for i in range(num_dets):
        det_standard = dets_standard[i]
        num_candidates = len(det_standard['candidates'])

        dets_CPN = []
        for j in range(num_candidates):
            det = {}
            det['image_id'] = det_standard['image']['id']
            det['bbox'] = det_standard['candidates'][j]['det_bbox']
            det['bbox_score'] = det_standard['candidates'][j]['det_score']
            det['imgpath'] = os.path.join(det_standard['image']['folder'], det_standard['image']['name'])
            if det['bbox_score'] >= bbox_thresh:
                dets_CPN.append(det)
        dets_CPN_list.append(dets_CPN)
    return dets_CPN_list


def get_bbox_from_gt(python_data_gt_dets, img_id, det_id):
    # get box detections
    det = np.zeros((1, 4), dtype=np.float32)
    bbox = np.asarray(python_data_gt_dets[img_id][det_id]['bbox'])
    det[0, :4] = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
    return det[0].tolist()


def get_bbox_from_keypoints(keypoints_python_data):
    #if keypoints_python_data == []:
    if keypoints_python_data == [] or keypoints_python_data == 45*[0]:
        return [0, 0, 2, 2]

    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(int(num_keypoints / 3)):
        x = keypoints_python_data[3 * keypoint_id]
        y = keypoints_python_data[3 * keypoint_id + 1]
        vis = keypoints_python_data[3 * keypoint_id + 2]
        if vis != 0 and vis!= 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    scale = enlarge_scale # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh


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


def inference_keypoints(pose_estimator, test_data):
    cls_dets = test_data["bbox"]
    # nms on the bboxes
    if flag_nms is True:
        cls_dets, keep = apply_nms(cls_dets, nms_method, nms_thresh)
        test_data = np.asarray(test_data)[keep]
        if len(keep) == 0:
            return -1
    else:
        test_data = [test_data]

    # crop and detect pose
    pose_heatmaps, details, cls_skeleton, crops, start_id, end_id = get_pose_from_bbox(pose_estimator, test_data, cfg)
    # get keypoint positions from pose
    keypoints = get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id)
    # dump results
    results = prepare_results(test_data[0], keypoints, cls_dets)
    return results


def apply_nms(cls_dets, nms_method, nms_thresh):
    # nms and filter
    keep = np.where((cls_dets[:, 4] >= min_scores) &
                    ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
    cls_dets = cls_dets[keep]
    if len(cls_dets) > 0:
        if nms_method == 'nms':
            keep = gpu_nms(cls_dets, nms_thresh)
        elif nms_method == 'soft':
            keep = cpu_soft_nms(np.ascontiguousarray(cls_dets, dtype=np.float32), method=2)
        else:
            assert False
    cls_dets = cls_dets[keep]
    return cls_dets, keep


def get_pose_from_bbox(pose_estimator, test_data, cfg):
    cls_skeleton = np.zeros((len(test_data), cfg.nr_skeleton, 3))
    crops = np.zeros((len(test_data), 4))

    batch_size = 32
    start_id = 0
    end_id = min(len(test_data), batch_size)

    test_imgs = []
    details = []
    for i in range(start_id, end_id):
        test_img, detail = Preprocessing(test_data[i], stage='test')
        test_imgs.append(test_img)
        details.append(detail)

    details = np.asarray(details)
    feed = test_imgs
    for i in range(end_id - start_id):
        ori_img = test_imgs[i][0].transpose(1, 2, 0)
        flip_img = cv2.flip(ori_img, 1)
        feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
    feed = np.vstack(feed)

    res = pose_estimator.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]
    res = res.transpose(0, 3, 1, 2)

    for i in range(end_id - start_id):
        fmp = res[end_id - start_id + i].transpose((1, 2, 0))
        fmp = cv2.flip(fmp, 1)
        fmp = list(fmp.transpose((2, 0, 1)))
        for (q, w) in cfg.symmetry:
            fmp[q], fmp[w] = fmp[w], fmp[q]
        fmp = np.array(fmp)
        res[i] += fmp
        res[i] /= 2
    pose_heatmaps = res
    return pose_heatmaps, details, cls_skeleton, crops, start_id, end_id


def get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id):
    res = pose_heatmaps
    for test_image_id in range(start_id, end_id):
        r0 = res[test_image_id - start_id].copy()
        r0 /= 255.
        r0 += 0.5

        for w in range(cfg.nr_skeleton):
            res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])

        border = 10
        dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy()

        for w in range(cfg.nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)

        for w in range(cfg.nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

        # map back to original images
        crops[test_image_id, :] = details[test_image_id - start_id, :]
        for w in range(cfg.nr_skeleton):
            cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
            cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
    return cls_skeleton


def prepare_results(test_data, cls_skeleton, cls_dets):
    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)

    cls_scores = 1
    dump_results = []
    cls_skeleton = np.concatenate(
        [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
        axis=1)
    for i in range(len(cls_skeleton)):
        result = dict(image_id=test_data['img_id'],
                      category_id=1,
                      score=float(round(cls_skeleton[i][-1], 4)),
                      keypoints=cls_skeleton[i][:-1].round(3).tolist())
        dump_results.append(result)
    return dump_results


def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False


def pose_to_standard_mot(keypoints_list_list, dets_list_list):
    openSVAI_python_data_list = []

    num_keypoints_list = len(keypoints_list_list)
    num_dets_list = len(dets_list_list)
    assert(num_keypoints_list == num_dets_list)

    for i in range(num_dets_list):

        dets_list = dets_list_list[i]
        keypoints_list = keypoints_list_list[i]

        if dets_list == []:
            continue
        img_path = dets_list[0]["imgpath"]
        img_folder_path = os.path.dirname(img_path)
        img_name =  os.path.basename(img_path)
        img_info = {"folder": img_folder_path,
                    "name": img_name,
                    "id": [int(i)]}
        openSVAI_python_data = {"image":[], "candidates":[]}
        openSVAI_python_data["image"] = img_info

        num_dets = len(dets_list)
        num_keypoints = len(keypoints_list) #number of persons, not number of keypoints for each person
        candidate_list = []

        for j in range(num_dets):
            keypoints_dict = keypoints_list[j]
            dets_dict = dets_list[j]

            img_id = keypoints_dict["img_id"]
            det_id = keypoints_dict["det_id"]
            track_id = keypoints_dict["track_id"]
            img_path = keypoints_dict["imgpath"]

            bbox_dets_data = dets_list[det_id]
            det = dets_dict["bbox"]
            if  det == [0, 0, 2, 2]:
                # do not provide keypoints
                candidate = {"det_bbox": [0, 0, 2, 2],
                             "det_score": 0}
            else:
                bbox_in_xywh = det[0:4]
                keypoints = keypoints_dict["keypoints"]

                track_score = sum(keypoints[2::3])/len(keypoints)/3.0

                candidate = {"det_bbox": bbox_in_xywh,
                             "det_score": 1,
                             "track_id": track_id,
                             "track_score": track_score,
                             "pose_keypoints_2d": keypoints}
            candidate_list.append(candidate)
        openSVAI_python_data["candidates"] = candidate_list
        openSVAI_python_data_list.append(openSVAI_python_data)
    return openSVAI_python_data_list


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def next_img_path(img_path):
    folder_path, img_name = os.path.split(img_path)
    img_name_no_ext =  img_name.split(".")[0]
    img_ext = img_name.split(".")[1]

    img_id = int(img_name_no_ext)
    next_img_id = img_id + 1

    if next_img_id <= 9:
        num_zeros = 5
    elif next_img_id <= 99:
        num_zeros = 4
    else:
        num_zeros = 3
    next_img_name = ""
    for i in range(num_zeros):
        next_img_name += "0"
    next_img_name += str(next_img_id)
    next_img_name += "."
    next_img_name += img_ext
    next_img_path = os.path.join(folder_path, next_img_name)
    return next_img_path


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    return False


if __name__ == '__main__':
    global args
    args = parse_args()

    initialize_parameters()

    # initialize pose estimator
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(args.test_model)

    if args.dataset_split == "posetrack18_val":
        image_folder = "data/Data_2018/posetrack_data/images/val/"
        if args.det_or_gt == "gt":
            detections_openSVAI_folder = "data/Data_2018/posetrack_data/annotations_openSVAI/"
        elif args.det_or_gt == "det":
            detections_openSVAI_folder = "data/Data_2018/posetrack_data/detections_openSVAI/"
        output_json_folder = "data/Data_2018/posetrack_results/lighttrack/results_openSVAI/"

    visualize_folder = "data/Data_2018/posetrack_results/lighttrack/visualize/"
    output_video_folder = "data/Data_2018/videos/"

    det_file_paths = get_immediate_childfile_paths(detections_openSVAI_folder)

    for det_file_path in det_file_paths:

        json_name = os.path.basename(det_file_path)
        output_json_path = os.path.join(output_json_folder, json_name)

        video_name = json_name.split(".")[0]
        image_subfolder = os.path.join(image_folder, video_name)

        visualize_subfolder = os.path.join(visualize_folder, video_name)
        output_video_path = os.path.join(output_video_folder, video_name+".mp4")

        light_track(pose_estimator,
                    det_file_path, output_json_path,
                    image_subfolder, visualize_subfolder, output_video_path)

        print("Finished video {}".format(output_video_path))
