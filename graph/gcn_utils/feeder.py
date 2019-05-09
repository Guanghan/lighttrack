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
                 data_neg_path,
                 ignore_empty_sample=True,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.neg_data_path = data_neg_path
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        with open(self.data_path, 'rb') as handle:
            self.graph_pos_pair_list_all = pickle.load(handle)


        with open(self.neg_data_path, 'rb') as handle:
            self.graph_neg_pair_list_all = pickle.load(handle)

        # output data shape (N, C, T, V, M)
        self.N = min(len(self.graph_pos_pair_list_all) , len(self.graph_neg_pair_list_all))  #sample
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
        if random_num > 0.5:
        #if False:
            # output shape (C, T, V, M)
            # get data
            sample_graph_pair = self.graph_pos_pair_list_all[index]
            label = 1 # a pair should match
        else:
            sample_graph_pair = self.graph_neg_pair_list_all[index]
            label = 0 # a pair does not match

        data_numpy_pair = []
        for siamese_id in range(2):
            # fill data_numpy
            data_numpy = np.zeros((self.C, self.T, self.V, 1))

            pose = sample_graph_pair[:][siamese_id]
            data_numpy[0, 0, :, 0] = [x[0] for x in pose]
            data_numpy[1, 0, :, 0] = [x[1] for x in pose]
            data_numpy_pair.append(data_numpy)

        return data_numpy_pair[0], data_numpy_pair[1], label
