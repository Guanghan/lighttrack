'''
    utils_io_file:
                  utilities for file-related I/O
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''
import os
from utils_io_folder import get_parent_folder_from_path, create_folder
import cv2

def find_file_ext(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    return file_extension


def validate_file_format(file_in_path, allowed_format):
    if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in allowed_format:
        return True
    else:
        return False


class Error(Exception):
    """Base class for other exceptions"""
    pass


class FormatIncorrectError(Error):
    """Raised when the file is of incorrect format"""
    pass


def is_image(file_in_path):
    if validate_file_format(file_in_path, ['jpg', 'JPEG', 'png', 'JPG']):
        return True
    else:
        return False


def is_video(file_in_path):
    if validate_file_format(file_in_path, ['avi', 'mkv', 'mp4']):
        return True
    else:
        return False


def file_to_img(filepath):
    try:
        img = cv2.imread(filepath)
    except IOError:
        print('cannot open image file: ' + filepath)
    else:
        print('unknown error reading image file')
    return img


def file_to_video(filepath):
    try:
        video = cv2.VideoCapture(filepath)
    except IOError:
        print('cannot open video file: ' + filepath)
    else:
        print('unknown error reading video file')
    return video


def video_to_images(video_file_path, output_img_folder_path = None):
    video = file_to_video(video_file_path)
    if output_img_folder_path is None:
        parent_folder_path, _ = get_parent_folder_from_path(video_file_path)
        video_name = os.path.basename(video_file_path)
        print(parent_folder_path)
        print(video_name)
        video_name_no_ext = os.path.splitext(video_name)[0]
        output_img_folder_path = os.path.join(parent_folder_path, video_name_no_ext)
    create_folder(output_img_folder_path)

    success, image = video.read()
    count = 0
    while success:
        img_name = "frame%05d.jpg" % count
        img_path = os.path.join(output_img_folder_path, img_name)
        cv2.imwrite(img_path, image)     # save frame as JPEG file
        success,image = video.read()
        #print('Read a new frame: ', success)
        count += 1
    return True

def count_lead_and_trail_zeroes(d):
    # https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightLinear
    if d:
        v = (d ^ (d - 1) >> 1)  # Set v's trailing 0s to 1s and zero rest
        trailing = 0
        while v:
            v >>= 1
            trailing += 1

        leading = 64
        v = d
        while v:
            v >>= 1
            leading -= 1
        return leading, trailing
    return 64, 64
