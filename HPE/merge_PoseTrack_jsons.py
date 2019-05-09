from utils_json import read_json_from_file, write_json_to_file
from utils_io_folder import get_immediate_childfile_paths

def merge_posetrack_jsons():
    posetrack_annotation_folder = "../data/Data_2017/posetrack_data/annotations/train"
    save_json_path = "posetrack_merged_train.json"
    gt_json_paths = get_immediate_childfile_paths(posetrack_annotation_folder, "json")
    merge_json(gt_json_paths, save_json_path)

    posetrack_annotation_folder = "../data/Data_2017/posetrack_data/annotations/val"
    save_json_path = "posetrack_merged_val.json"
    gt_json_paths = get_immediate_childfile_paths(posetrack_annotation_folder, "json")
    merge_json(gt_json_paths, save_json_path)
    return


def merge_json(gt_json_paths, save_json_path):
    python_data_merged = {"annolist": []}
    for gt_json_path in gt_json_paths:
        python_data = read_json_from_file(gt_json_path)
        python_data_merged["annolist"].extend(python_data["annolist"])
    write_json_to_file(python_data_merged, save_json_path, flag_verbose = False)


if __name__ == "__main__":
    merge_posetrack_jsons()
