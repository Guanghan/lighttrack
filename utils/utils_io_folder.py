'''
    utils_io_folder:
                    utilities for folder-related I/O
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''
import os
from utils_natural_sort import natural_sort

def get_immediate_subfolder_paths(folder_path):
    subfolder_names = get_immediate_subfolder_names(folder_path)
    subfolder_paths = [os.path.join(folder_path, subfolder_name) for subfolder_name in subfolder_names]
    return subfolder_paths


def get_immediate_subfolder_names(folder_path):
    subfolder_names = [folder_name for folder_name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, folder_name))]
    natural_sort(subfolder_names)
    return subfolder_names

def get_immediate_childfile_paths(folder_path, ext = None, exclude = None):
    files_names = get_immediate_childfile_names(folder_path, ext, exclude)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def get_immediate_childfile_names(folder_path, ext = None, exclude = None):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    if ext is not None:
        files_names = [file_name for file_name in files_names
                       if file_name.endswith(ext)]
    if exclude is not None:
        files_names = [file_name for file_name in files_names
                       if not file_name.endswith(exclude)]
    natural_sort(files_names)
    return files_names


def get_folder_name_from_path(folder_path):
    path, folder_name = os.path.split(folder_path)
    return folder_name


def get_parent_folder_from_path(folder_path):
    parent_folder_path = os.path.abspath(os.path.join(folder_path, os.pardir))
    parent_folder_name = os.path.basename(parent_folder_path)
    return parent_folder_path, parent_folder_name


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
