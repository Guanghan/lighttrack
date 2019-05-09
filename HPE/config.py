import os
import os.path as osp
import sys
import numpy as np

#dataset_name = "COCO"
dataset_name = "posetrack"
#dataset_name = "posetrack+COCO"

class Config:
    username = 'default'

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    proj_name = this_dir_name

    bbox_thresh = 0.4

    # output path
    batch_size = 36
    if dataset_name == "COCO":
        output_dir = os.path.join(root_dir, 'logs', username + '.' + this_dir_name)
        model_dump_dir = osp.join(output_dir, 'model_dump_COCO')
        epoch_size = int(149025 / batch_size) * 10
    elif dataset_name == "posetrack":
        output_dir = os.path.join(root_dir, 'logs', username + '.' + this_dir_name)
        model_dump_dir = osp.join(output_dir, 'model_dump_PT')
        epoch_size = int(51136 / batch_size) * 10
    elif dataset_name == "posetrack+COCO":
        output_dir = os.path.join(root_dir, 'logs', username + '.' + this_dir_name)
        model_dump_dir = osp.join(output_dir, 'model_dump_PTCOCO')
        epoch_size = int(200161 / batch_size) * 10

    display = 1

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = 60

    optimizer = 'adam'

    weight_decay = 1e-5

    step_size = epoch_size * lr_dec_epoch
    max_itr = epoch_size * 400
    double_bias = False

    dpflow_enable = True
    nr_dpflows = 10

    gpu_ids = '0'
    nr_gpus = 1
    continue_train = True

    def get_lr(self, itr):
        lr = self.lr * self.lr_gamma ** (itr // self.step_size)
        return lr

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.nr_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))

    bn_train = True
    init_model = osp.join(root_dir, 'data', 'imagenet_weights', 'res101.ckpt')

    if dataset_name == "COCO":
        nr_skeleton = 17
        img_path = os.path.join(root_dir, 'data', 'COCO', 'MSCOCO', 'images')
        symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
        pixel_norm = True

    elif dataset_name == "posetrack" or dataset_name == "posetrack+COCO":
        nr_skeleton = 15
        img_path = os.path.join(root_dir, 'data', 'Data_2018', 'posetrack_data')
        symmetry = [(0, 5), (1, 4), (2, 3), (6, 11), (7, 10), (8, 9)]

        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
        pixel_norm = True

    imgExtXBorder = 0.1
    imgExtYBorder = 0.15
    min_kps = 1

    use_seg = False

    data_aug = True # has to be true
    nr_aug = 4

    data_shape = (384, 288) #height, width
    output_shape = (96, 72) #height, width
    gaussain_kernel = (13, 13)

    gk15 = (23, 23)
    gk11 = (17, 17)
    gk9 = (13, 13)
    gk7 = (9, 9)

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'lib'))
from tfflat.utils import add_pypath, make_link, make_dir
add_pypath(osp.join(cfg.root_dir, 'data'))
add_pypath(osp.join(cfg.root_dir, 'data', 'COCO'))

make_link(cfg.output_dir, './log')
make_dir(cfg.output_dir)
make_dir(cfg.model_dump_dir)
