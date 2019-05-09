'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    July 2nd, 2018
'''
import sys, os
sys.path.append(os.path.abspath("utils/"))

from utils_json import *
from utils_io_folder import *
import argparse

dataset_splits = ['whole', 'val', 'test']
dataset_split = "val"

input_keypoints_format = "PoseTrack"

PoseTrack_data = {"annolist":
        [{"image": [{"name": "/export/guanghan/Data/posetrack_data/images/bonn_5sec/020910_mpii/00000001.jpg"}],
          "annorect": [
                      {"y2": [820], "annopoints": [{"point": [{"y": [480.276], "x": [1309.639], "score": [1.0], "id": [0]}, {"y": [471.052], "x": [1308.319], "score": [1.0], "id": [1]}, {"y": [472.37], "x": [1309.639], "score": [1.0], "id": [2]}, {"y": [456.557], "x": [1267.417], "score": [1.0], "id": [3]}, {"y": [476.323], "x": [1300.403], "score": [1.0], "id": [4]}, {"y": [480.276], "x": [1225.194], "score": [1.0], "id": [5]}, {"y": [564.609], "x": [1299.083], "score": [1.0], "id": [6]}, {"y": [550.115], "x": [1126.236], "score": [1.0], "id": [7]}, {"y": [575.151], "x": [1151.306], "score": [1.0], "id": [8]}, {"y": [602.823], "x": [1229.153], "score": [1.0], "id": [9]}, {"y": [676.615], "x": [1226.514], "score": [1.0], "id": [10]}, {"y": [706.922], "x": [1188.25], "score": [1.0], "id": [11]}, {"y": [592.281], "x": [1292.486], "score": [1.0], "id": [12]}, {"y": [577.1275], "x": [1203.4235], "score": [1.0], "id": [13]}, {"y": [561.974], "x": [1114.361], "score": [1.0], "id": [14]}]}], "track_id": [0], "y1": [423], "score": [0.9997325539588928], "x2": [1329], "x1": [1094]},
                      {"y2": [940], "annopoints": [{"point": [{"y": [599.656], "x": [1084.479], "score": [1.0], "id": [0]}, {"y": [589.703], "x": [1085.903], "score": [1.0], "id": [1]}, {"y": [589.703], "x": [1085.903], "score": [1.0], "id": [2]}, {"y": [569.797], "x": [1034.653], "score": [1.0], "id": [3]}, {"y": [593.969], "x": [1078.785], "score": [1.0], "id": [4]}, {"y": [599.656], "x": [999.062], "score": [1.0], "id": [5]}, {"y": [770.281], "x": [1041.771], "score": [1.0], "id": [6]}, {"y": [714.828], "x": [892.292], "score": [1.0], "id": [7]}, {"y": [724.781], "x": [936.424], "score": [1.0], "id": [8]}, {"y": [815.781], "x": [896.562], "score": [1.0], "id": [9]}, {"y": [800.141], "x": [1028.958], "score": [1.0], "id": [10]}, {"y": [844.219], "x": [822.535], "score": [1.0], "id": [11]}, {"y": [719.094], "x": [1036.076], "score": [1.0], "id": [12]}, {"y": [726.914], "x": [1018.281], "score": [1.0], "id": [13]}, {"y": [734.734], "x": [1000.486], "score": [1.0], "id": [14]}]}], "track_id": [1], "y1": [536], "score": [0.9994370341300964], "x2": [1112], "x1": [796]},
                      {"y2": [742], "annopoints": [{"point": [{"y": [397.156], "x": [848.719], "score": [1.0], "id": [0]}, {"y": [389.474], "x": [850.257], "score": [1.0], "id": [1]}, {"y": [391.01], "x": [848.719], "score": [1.0], "id": [2]}, {"y": [371.036], "x": [807.188], "score": [1.0], "id": [3]}, {"y": [397.156], "x": [836.413], "score": [1.0], "id": [4]}, {"y": [386.401], "x": [757.965], "score": [1.0], "id": [5]}, {"y": [470.906], "x": [842.566], "score": [1.0], "id": [6]}, {"y": [484.734], "x": [677.979], "score": [1.0], "id": [7]}, {"y": [507.781], "x": [691.823], "score": [1.0], "id": [8]}, {"y": [526.219], "x": [762.58], "score": [1.0], "id": [9]}, {"y": [604.578], "x": [751.812], "score": [1.0], "id": [10]}, {"y": [619.943], "x": [739.507], "score": [1.0], "id": [11]}, {"y": [501.635], "x": [824.108], "score": [1.0], "id": [12]}, {"y": [451.70050000000003], "x": [729.509], "score": [1.0], "id": [13]}, {"y": [401.766], "x": [634.91], "score": [1.0], "id": [14]}]}], "track_id": [2], "y1": [337], "score": [0.9968172311782837], "x2": [871], "x1": [613]},
                      {"y2": [601], "annopoints": [{"point": [{"y": [258.724], "x": [975.375], "score": [1.0], "id": [0]}, {"y": [252.409], "x": [976.639], "score": [1.0], "id": [1]}, {"y": [252.409], "x": [976.639], "score": [1.0], "id": [2]}, {"y": [238.516], "x": [939.986], "score": [1.0], "id": [3]}, {"y": [252.409], "x": [961.472], "score": [1.0], "id": [4]}, {"y": [253.672], "x": [886.903], "score": [1.0], "id": [5]}, {"y": [369.87], "x": [970.319], "score": [1.0], "id": [6]}, {"y": [344.609], "x": [818.653], "score": [1.0], "id": [7]}, {"y": [364.818], "x": [864.153], "score": [1.0], "id": [8]}, {"y": [429.232], "x": [880.583], "score": [1.0], "id": [9]}, {"y": [460.807], "x": [917.236], "score": [1.0], "id": [10]}, {"y": [511.328], "x": [866.681], "score": [1.0], "id": [11]}, {"y": [368.607], "x": [946.306], "score": [1.0], "id": [12]}, {"y": [345.8725], "x": [870.4725000000001], "score": [1.0], "id": [13]}, {"y": [323.138], "x": [794.639], "score": [1.0], "id": [14]}]}], "track_id": [3], "y1": [205], "score": [0.980134904384613], "x2": [994], "x1": [776]}   ]
                   }]
}

def standard_to_PoseTrack_18(standard_keypoints_ret, gt_python_data, mode_track = True, bbox_thresh = 0):
    PoseTrack_dict = {"images": [],
                      "annotations": [],
                      "categories": [{}]}
    PoseTrack_dict["categories"][0]["name"] = "person"
    PoseTrack_dict["categories"][0]["keypoints"] = ['right_ankle', 'right_knee', 'right_hip',
                                                'left_hip', 'left_knee', 'left_ankle',
                                                'right_wrist', 'right_elbow', 'right_shoulder',
                                                'left_shoulder', 'left_elbow', 'left_wrist',
                                                'head_bottom', 'nose', 'head_top'] #PoseTrack2017 pose order
    PoseTrack_images_info_list = []
    PoseTrack_annotations_info_list = []

    for standard_data_item in standard_keypoints_ret:
        image_name = standard_data_item["image"]["name"]
        folder_name = os.path.basename(standard_data_item["image"]["folder"])
        _, parent_folder_name = get_parent_folder_from_path(standard_data_item["image"]["folder"])
        img_path = os.path.join("images", parent_folder_name, folder_name, image_name)

        print(img_path)

        gt_images_info = gt_python_data["images"]
        frame_id = find_id_from_annotation_by_name(gt_images_info, img_path)
        PoseTrack_images_info_item = {"file_name": img_path,
                                             "id": frame_id}
        PoseTrack_images_info_list.append(PoseTrack_images_info_item)

        candidates = standard_data_item["candidates"]
        for candidate in candidates:
            det_score = candidate["det_score"]
            if det_score < args.bbox_thresh: continue
            if "pose_keypoints_2d" not in candidate: continue

            if mode_track is True:
                track_id = candidate["track_id"]
            else:
                track_id = -1
            keypoints = candidate["pose_keypoints_2d"]
            scores = candidate["pose_keypoints_2d"][2::3]
            PoseTrack_annotations_info_item = {"image_id": frame_id,
                                               "track_id": track_id,
                                               "keypoints": keypoints,
                                               "scores": scores }
                                               #"score": scores }
            PoseTrack_annotations_info_list.append(PoseTrack_annotations_info_item)

    PoseTrack_dict["images"] = PoseTrack_images_info_list
    PoseTrack_dict["annotations"] = PoseTrack_annotations_info_list
    return PoseTrack_dict


# PoseFlow might output the format exactly like the format required by the PoseTrack dataset, therefore we do not need to do any conversion.
# If we only output [detection + pose estimation], then we need the conversion from standard openSVAI into PoseTrack format.
def standard_to_PoseTrack_17(standard_keypoints_ret, gt_python_data, mode_track = True, bbox_thresh = 0, drop_thresh = 0.8):
    PoseTrack_data_content_list = []

    for standard_data_item in standard_keypoints_ret:
        image_name = standard_data_item["image"]["name"]
        folder_name = os.path.basename(standard_data_item["image"]["folder"])
        _, parent_folder_name = get_parent_folder_from_path(standard_data_item["image"]["folder"])
        img_path = os.path.join("images", parent_folder_name, folder_name, image_name)

        PoseTrack_data_content = {}
        PoseTrack_data_content["image"] = [{"name": img_path}]

        ''' check if this is within gt '''
        gt_images_info = gt_python_data["images"]
        frame_id, index = find_id_from_annotation_by_name(gt_images_info, img_path)

        annorect = []
        standard_data_item_candidates = standard_data_item["candidates"]
        for standard_data_item_candidate in standard_data_item_candidates:
            det_bbox = standard_data_item_candidate["det_bbox"]
            det_score = standard_data_item_candidate["det_score"]

            if det_score < bbox_thresh: continue
            if "pose_keypoints_2d" not in standard_data_item_candidate: continue
            pose_keypoints_2d = standard_data_item_candidate["pose_keypoints_2d"]

            if mode_track is True:
                track_id = standard_data_item_candidate["track_id"]
                track_score = standard_data_item_candidate["track_score"]

            PoseTrack_data_content_candidate = {}
            PoseTrack_data_content_candidate["x1"] = [det_bbox[0]]
            PoseTrack_data_content_candidate["y1"] = [det_bbox[1]]
            PoseTrack_data_content_candidate["x2"] = [det_bbox[0] + det_bbox[2]]
            PoseTrack_data_content_candidate["y2"] = [det_bbox[1] + det_bbox[3]]
            if mode_track is True:
                PoseTrack_data_content_candidate["track_id"] = [track_id]
                PoseTrack_data_content_candidate["score"] = [track_score]
            else:
                PoseTrack_data_content_candidate["track_id"] = [-1]
                PoseTrack_data_content_candidate["score"] = [0]

            annopoints_dict = []
            num_keypoints = int(len(pose_keypoints_2d)/3)

            if input_keypoints_format == "PoseTrack":
                pose_keypoints_2d_PoseTrack = pose_keypoints_2d

            num_keypoints_PoseTrack = int(len(pose_keypoints_2d_PoseTrack)/3)

            for i in range(num_keypoints_PoseTrack):
                annopoint = {}
                annopoint["x"] = [pose_keypoints_2d_PoseTrack[3*i]]
                annopoint["y"] =  [pose_keypoints_2d_PoseTrack[3*i+1]]
                annopoint["score"] = [pose_keypoints_2d_PoseTrack[3*i+2]]

                # Drop keypoints based on the corresponding confidence
                if annopoint["score"][0] <= drop_thresh:
                    continue

                annopoint["id"] = [i]
                annopoints_dict.append(annopoint)
            annopoints = [{"point": annopoints_dict}]
            PoseTrack_data_content_candidate["annopoints"] = annopoints

            annorect.append(PoseTrack_data_content_candidate)
        PoseTrack_data_content["annorect"] = annorect
        PoseTrack_data_content_list.append(PoseTrack_data_content)
    PoseTrack_data["annolist"] = PoseTrack_data_content_list
    return PoseTrack_data


def batch_standard_to_PoseTrack_17(dataset_split = "light_track", mode = "pose", bbox_thresh = 0, drop_thresh = 0):
    if dataset_split == "light_track":
        gt_json_folder_base = "data/Data_2018/posetrack_data/annotations/val"
        input_json_folder_base = "data/Data_2018/posetrack_results/lighttrack/results_openSVAI"
        output_json_folder_base = "data/Data_2018/predictions_lighttrack/"

    gt_json_file_paths = get_immediate_childfile_paths(gt_json_folder_base, ext=".json")
    for gt_json_file_path in gt_json_file_paths:
        json_file_name = os.path.basename(gt_json_file_path)

        input_json_file_path = os.path.join(input_json_folder_base, json_file_name)
        output_json_file_path = os.path.join(output_json_folder_base, json_file_name)
        print("Reading Json: ", input_json_file_path)

        rets_video_standard = read_json_from_file(input_json_file_path)
        gt_python_data = read_json_from_file(gt_json_file_path)

        if mode == "pose":
            rets_video_posetrack_17 = standard_to_PoseTrack_17(rets_video_standard, gt_python_data, False, bbox_thresh, drop_thresh)
        elif mode == "track":
            rets_video_posetrack_17 = standard_to_PoseTrack_17(rets_video_standard, gt_python_data, True, bbox_thresh, drop_thresh)

        write_json_to_file(rets_video_posetrack_17, output_json_file_path, flag_verbose = False)
    return


def batch_standard_to_PoseTrack_18(dataset_split = "val", mode = "pose", bbox_thresh = 0):
    if dataset_split == "light_track":
        input_json_folder_base = "data/Data_2018/posetrack_results/lighttrack/results_openSVAI"
        gt_json_folder_base = "data/Data_2018/posetrack_data/annotations/val"
        output_json_folder_base = "data/Data_2018/predictions_lighttrack/"

    gt_json_file_paths = get_immediate_childfile_paths(gt_json_folder_base, ext=".json")
    for gt_json_file_path in gt_json_file_paths:
        json_file_name = os.path.basename(gt_json_file_path)
        input_json_file_path = os.path.join(input_json_folder_base, json_file_name)
        output_json_file_path = os.path.join(output_json_folder_base, json_file_name)
        print("Reading Json: ", input_json_file_path)

        rets_video_standard = read_json_from_file(input_json_file_path)
        gt_python_data = read_json_from_file(gt_json_file_path)

        if mode == "pose":
            rets_video_posetrack_18 = standard_to_PoseTrack_18(rets_video_standard, gt_python_data, False, bbox_thresh)
        elif mode == "track":
            rets_video_posetrack_18 = standard_to_PoseTrack_18(rets_video_standard, gt_python_data, True, bbox_thresh)

        write_json_to_file(rets_video_posetrack_18, output_json_file_path, flag_verbose = False)
    return


def find_id_from_annotation_by_name(gt_images_info, img_path):
    index_list = find(gt_images_info, key="file_name", value=img_path)
    assert(len(index_list) >= 1)
    index = index_list[0]
    frame_id = gt_images_info[index]["frame_id"]
    return frame_id, index


def find(lst, key, value):
    # find the index of a dict in list
    index_list = []
    for i, dic in enumerate(lst):
        if dic[key] == value:
            index_list.append(i)
    return index_list


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--bbox_thresh', '-e', type=float, dest='bbox_thresh', default = 0)
        parser.add_argument('--drop_thresh', '-r', type=float, dest='drop_thresh', default = 0)
        parser.add_argument('--mode', '-m', type=str, dest='mode', default = "pose")
        parser.add_argument('--dataset_split', '-d', type=str, dest='dataset_split', default = "val")
        parser.add_argument('--format', '-f', type=str, dest='format', default = "17")
        args = parser.parse_args()
        return args
    global args
    args = parse_args()
    print("Using detection threshold: ", args.bbox_thresh)

    # The following output formats (17 and 18) should have identical evaluation results
    # PoseTrack'18 format is designed such that it is easily compatible with COCO
    # During evaluation, it seems that PoseTrack'18 format json will be transformed back to PoseTrack'17 format,
    # Therefore, it is okay to just output in PoseTrack'17 format.
    if args.format == "17":
        # Generate PoseTrack17 format jsons for quantitative evaluation
        batch_standard_to_PoseTrack_17("light_track",
                                       args.mode,
                                       args.bbox_thresh,
                                       args.drop_thresh)

    elif args.format == "18":
        # Generate PoseTrack18 format jsons for quantitative evaluation
        batch_standard_to_PoseTrack_18(args.dataset_split,
                                       args.mode,
                                       args.bbox_thresh)
