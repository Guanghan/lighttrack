'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 24th, 2018

    Feeder of Siamese Graph Convolutional Networks for Pose Tracking
    Code partially borrowed from:
    https://github.com/yysijie/st-gcn/blob/master/feeder/feeder.py
'''
# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
from . import tools
import random

class Feeder(torch.utils.data.Dataset):
    """ Feeder of PoseTrack Dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 ignore_empty_sample=True,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.data_path_triplet = data_path  # use triplet loss
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        with open(self.data_path_triplet, 'rb') as handle:
            self.graph_triplet_list_all = pickle.load(handle)

        # output data shape (N, C, T, V, M)
        self.N = len(self.graph_triplet_list_all)  #sample
        self.C = 2  #channel
        self.T = 1  #frame
        self.V = 15  #joint
        self.M = 1  #person

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # randomly add negative samples
        random_num = random.uniform(0, 1)

        sample_graph_triplet = self.graph_triplet_list_all[index]

        # fill data_numpy
        anchor = np.zeros((self.C, self.T, self.V, 1))
        pos = np.zeros((self.C, self.T, self.V, 1))
        neg = np.zeros((self.C, self.T, self.V, 1))

        #print(sample_graph_triplet[:])

        pose1 = sample_graph_triplet[:][0]
        anchor[0, 0, :, 0] = [x[0] for x in pose1]
        anchor[1, 0, :, 0] = [x[1] for x in pose1]

        pose2 = sample_graph_triplet[:][1]
        pos[0, 0, :, 0] = [x[0] for x in pose2]
        pos[1, 0, :, 0] = [x[1] for x in pose2]

        pose3 = sample_graph_triplet[:][2]
        neg[0, 0, :, 0] = [x[0] for x in pose3]
        neg[1, 0, :, 0] = [x[1] for x in pose3]

        #print("anchor: {}".format(pose1))
        #print("pos: {}".format(pose2))
        #print("neg: {}".format(pose3))

        return anchor, pos, neg
