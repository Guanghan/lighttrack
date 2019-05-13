# largely borrowed from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/a68d786f6c9cb65d944c2f48eb7d219c914de11f/detect.py
from __future__ import division

from detector.models import *
from detector.detector_utils import *

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="detector/config/yolov3.cfg", help="path to model config file")
parser.add_argument("--weights_path", type=str, default="weights/YOLOv3/yolov3.weights", help="path to weights file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
print("Detector YOLOv3 options:", opt)

cuda = torch.cuda.is_available()

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

if cuda:
    model.cuda()

model.eval()  # Set in evaluation mode


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index


def inference_yolov3(img_path):
    img = np.array(Image.open(img_path))
    return inference_yolov3_from_img(img)


def inference_yolov3_from_img(img):
    input_img = preprocess_img_for_yolo(img)

    # Configure input
    input_img = Variable(input_img.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
        if detections is None:
            return []
        else:
            detections = detections.data.cpu().numpy()

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    human_candidates = []
    if detections is not None:
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            if int(cls_pred) == 0:
                human_candidate = [x1, y1, box_w, box_h]
                human_candidates.append(human_candidate)
    return human_candidates


if __name__ == "__main__":
    img_path = "/export/guanghan/PyTorch-YOLOv3/data/samples/messi.jpg"
    human_candidates = inference_yolov3(img_path)
    print("human_candidates:", human_candidates)
