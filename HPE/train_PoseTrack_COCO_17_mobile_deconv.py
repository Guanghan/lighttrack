'''
    Modified by: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    March, 2019
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys, os
import argparse
import numpy as np
from functools import partial

from config import cfg
from tfflat.base import ModelDesc, Trainer
from tfflat.utils import mem_info

from nets.mobilenet_v1 import mobilenet_v1_base, mobilenet_v1_arg_scope

class Network(ModelDesc):

    def render_gaussian_heatmap(self, coord, output_shape, sigma):

        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))

        x = tf.floor(tf.reshape(coord[:,:,0],[-1,1,1,cfg.nr_skeleton]) / cfg.data_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:,:,1],[-1,1,1,cfg.nr_skeleton]) / cfg.data_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))
        return heatmap * 255.

    def make_data(self):
        ''' Load PoseTrack data '''
        from AllJoints_PoseTrack import PoseTrackJoints
        from AllJoints_COCO import PoseTrackJoints_COCO
        from dataset import Preprocessing

        d = PoseTrackJoints()
        train_data, _ = d.load_data(cfg.min_kps)
        print(len(train_data))

        #'''
        d2 = PoseTrackJoints_COCO()
        train_data_coco, _ = d2.load_data(cfg.min_kps)
        print(len(train_data_coco))

        train_data.extend(train_data_coco)
        print(len(train_data))
        #'''

        from random import shuffle
        shuffle(train_data)

        from tfflat.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        dp = DataFromList(train_data)
        if cfg.dpflow_enable:
            dp = MultiProcessMapDataZMQ(dp, cfg.nr_dpflows, Preprocessing)
        else:
            dp = MapData(dp, Preprocessing)
        dp = BatchData(dp, cfg.batch_size // cfg.nr_aug)
        dp.reset_state()
        dataiter = dp.get_data()

        return dataiter

    def head_net(self, blocks, is_training, trainable=True):

        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):

            out = slim.conv2d_transpose(blocks, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up2')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up3')

            out = slim.conv2d(out, cfg.nr_skeleton, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')

        return out


    def make_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.data_shape, 3])

            label15 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label11 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label9 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label7 = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            valids = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.nr_skeleton])
            labels = [label15, label11, label9, label7]
            self.set_inputs(image, label15, label11, label9, label7, valids)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.data_shape, 3])
            self.set_inputs(image)

        mobilenet_v1_fms, endpoints = mobilenet_v1_base(image)

        heatmap_outs = self.head_net(mobilenet_v1_fms, is_train)

        # make loss
        if is_train:
            def ohkm(loss, top_k):
                ohkm_loss = 0.
                for i in range(cfg.batch_size):
                    sub_loss = loss[i]
                    topk_val, topk_idx = tf.nn.top_k(sub_loss, k=top_k, sorted=False, name='ohkm{}'.format(i))
                    tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
                    ohkm_loss += tf.reduce_sum(tmp_loss) / top_k
                ohkm_loss /= cfg.batch_size
                return ohkm_loss

            label = label7 * tf.to_float(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.nr_skeleton)), 0.1))
            loss = tf.reduce_mean(tf.square(heatmap_outs - label))

            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', '-d', type=str, dest='gpu_ids')
        parser.add_argument('--continue', '-c', dest='continue_train', action='store_true')
        parser.add_argument('--debug', dest='debug', action='store_true')
        args = parser.parse_args()

        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

        return args
    args = parse_args()

    cfg.set_args(args.gpu_ids, args.continue_train)
    trainer = Trainer(Network(), cfg)
    trainer.train()
