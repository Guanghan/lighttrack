'''
Json-related utilities for standardization of data

    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    June, 2018
'''

import json


def write_json_to_file(python_data, output_path, flag_verbose = False):
    with open(output_path, "w") as write_file:
        json.dump(python_data, write_file)
    if flag_verbose is True:
        print("Json string dumped to: %s", output_path)


def read_json_from_file(input_path):
    with open(input_path, "r") as read_file:
        python_data = json.load(read_file)
    return python_data


def python_to_json(python_data):
    '''Convert python data (tuple, list, dict, etc) into json string'''
    json_str = json.dumps(python_data, indent = 4)
    return json_str


def json_to_python(json_str):
    '''Convert json string to python data (tuple, list, dict, etc)'''
    python_data = json.loads(json_str)
    return python_data
