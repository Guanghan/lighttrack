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
                 #label_path,
                 ignore_empty_sample=True,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        with open(self.data_path, 'rb') as handle:
            self.graph_pair_list_all = pickle.load(handle)

        # output data shape (N, C, T, V, M)
        self.N = len(self.graph_pair_list_all)  #sample
        self.C = 2  #channel
        self.T = 1  #frame
        self.V = 15  #joint
        self.M = 1  #person

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_graph_pair = self.graph_pair_list_all[index]

        data_numpy_pair = []
        for siamese_id in range(2):
            # fill data_numpy
            data_numpy = np.zeros((self.C, self.T, self.V, 1))

            pose = sample_graph_pair[:][siamese_id]
            data_numpy[0, 0, :, 0] = [x[0] for x in pose]
            data_numpy[1, 0, :, 0] = [x[1] for x in pose]
            data_numpy_pair.append(data_numpy)

            # add label
            if siamese_id == 1:
                # positive sample
                label = 1 # a pair should match

                # randomly add negative samples
                if random.uniform(0, 1) > 0.5:
                    #self.change_pose_graph(data_numpy_pair[0])
                    self.change_pose_graph_debug(data_numpy_pair[0])
                    label = 0 # mis-match

        return data_numpy_pair[0], data_numpy_pair[1], label


    def change_pose_graph_debug(self, data_numpy):
        data_numpy[0, 0, :, 0] = 0
        data_numpy[1, 0, :, 0] = 0
        return


    def change_pose_graph(self, data_numpy):
        # change pose B so that pose A and pose B does not match
        pose = np.zeros((15, 2))
        pose[:, 0] = data_numpy[0, 0, :, 0]
        pose[:, 1] = data_numpy[1, 0, :, 0]

        for joint_id in range(15):
            offset_x = random.uniform(-0.2, 0.2)
            offset_y = random.uniform(-0.2, 0.2)

            pose[joint_id, 0] += offset_x
            pose[joint_id, 1] += offset_y

            for dim in range(2):
                pose[joint_id, dim] = min(pose[joint_id, dim], 1)
                pose[joint_id, dim] = max(pose[joint_id, dim], 0)

        data_numpy[0, 0, :, 0] = pose[:, 0]
        data_numpy[1, 0, :, 0] = pose[:, 1]
        return


    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
