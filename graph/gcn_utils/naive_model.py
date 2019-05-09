'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    November 8th, 2018

    Naive model: only use L2 distance for pose matching
    Compare with GCN.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    r"""Naive Model: only use L2 distance for pose matching

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()


    def forward(self, input_1, input_2):  # siamese network needs two times of forwards
        feature_1 = self.extract_feature(input_1)
        feature_2 = self.extract_feature(input_2)
        return feature_1, feature_2


    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.view(N, -1)

        # prediction
        feature = x.view(x.size(0), -1)

        return feature
