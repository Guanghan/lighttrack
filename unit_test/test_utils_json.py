import sys, os, io, shutil
sys.path.append(os.path.abspath("../utility/"))

from utils_json import *


def test_write_json_to_file():
    python_data =  {
        "version": "1.0",
    	"image": [
    	       {
    		      "folder": "images/bonn_5sec/000342_mpii",
    		      "name": "00000001.jpg",
                  "id" : 0,
    	       }
        ],
        "candidates":[
            {
              "det_category" : 1,
              "det_bbox" : [300,300,100,100],
              "det_score" : [0.9],

              "pose_order" : [1,2,3],
              "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

    		  "track_id": [0],
    		  "track_score": [0.8],
            },
            {
              "det_category" : 2,
              "det_bbox" : [300,300,100,100],
              "det_score" : [0.1],

              "pose_order" : [1,2,3],
              "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

    		  "track_id": [1],
    		  "track_score": [0.6],
            }
         ]
    }
    python_data_total = {}
    python_data_total.update(python_data)
    python_data_total.update(python_data)

    output_path = "temp.json"
    write_json_to_file(python_data_total, output_path, flag_verbose=True)
    flag_finished = True

    if flag_finished is not None:
        return True
    else:
        return False


def test_read_json_from_file():
    input_path = "temp.json"
    python_data = read_json_from_file(input_path)
    print(python_data["version"])
    print(python_data["candidates"][0]["det_bbox"])
    if python_data["version"] == "1.0":
        return True
    else:
        return False


def test_python_to_json():
    python_data =  {
        "version": "1.0",
    	"image": [
    	       {
    		      "folder": "images/bonn_5sec/000342_mpii",
    		      "name": "00000001.jpg",
                  "id" : 0,
    	       }
        ],
        "candidates":[
            {
              "det_category" : 1,
              "det_bbox" : [300,300,100,100],
              "det_score" : [0.9],

              "pose_order" : [1,2,3],
              "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

    		  "track_id": [0],
    		  "track_score": [0.8],
            },
            {
              "det_category" : 2,
              "det_bbox" : [300,300,100,100],
              "det_score" : [0.1],

              "pose_order" : [1,2,3],
              "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

    		  "track_id": [1],
    		  "track_score": [0.6],
            }
         ]
    }
    json_str = python_to_json(python_data)
    print(json_str)
    flag_finished = True

    if flag_finished is not None:
        return True
    else:
        return False



def test_json_to_python():
    python_data =  {
        "version": "1.0",
    	"image": [
    	       {
    		      "folder": "images/bonn_5sec/000342_mpii",
    		      "name": "00000001.jpg",
                  "id" : 0,
    	       }
        ],
        "candidates":[
            {
              "det_category" : 1,
              "det_bbox" : [300,300,100,100],
              "det_score" : [0.9],

              "pose_order" : [1,2,3],
              "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

    		  "track_id": [0],
    		  "track_score": [0.8],
            },
            {
              "det_category" : 2,
              "det_bbox" : [300,300,100,100],
              "det_score" : [0.1],

              "pose_order" : [1,2,3],
              "pose_keypoints_2d" : [10,10,0.9, 20,20,0.9, 30,30,0.8],

    		  "track_id": [1],
    		  "track_score": [0.6],
            }
         ]
    }
    json_str = python_to_json(python_data)
    python_data = json_to_python(json_str)
    print(python_data)
    json_str = python_to_json(python_data)
    print(json_str)

    flag_finished = True

    if flag_finished is not None:
        return True
    else:
        return False


def main():
    print("Testing: utils_json")

    passed = test_write_json_to_file()
    if passed is False:
        print("\t write_json_to_file failed")

    passed = test_read_json_from_file()
    if passed is False:
        print("\t read_json_from_file failed")

    passed = test_python_to_json()
    if passed is False:
        print("\t test_python_to_json failed")

    passed = test_json_to_python()
    if passed is False:
        print("\t test_json_to_python failed")

if __name__ == '__main__':
    main()
