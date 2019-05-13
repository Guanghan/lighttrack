'''
 Main utilities for Human Pose Estimation
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''

import sys, os, time, math, copy
import matplotlib.pyplot as plt

caffe_root = '/home/ngh/dev/Github/caffe-GNet/'
sys.path.insert(0, caffe_root + 'python')

sys.path.append(os.path.abspath("../utils/"))
from utils_io_file import is_image
from utils_io_folder import *
from utils_nms import find_joints_in_heatmaps_nms
from utils_convert_heatmap import *

heatmap_size = 64

# Choose what to display, if any
flag_demo_poses = True
flag_demo_heatmaps = False
flag_selective = False
IMG_NAMES_TO_SHOW= ['im1097.jpg']

# Choose what and how to draw joints and connections
flag_only_draw_sure = False
flag_color_sticks = True

''' ----------------------Pre-processing images------------------------------'''
def preprocess(img, mean):
    img_out = np.double(img)/255.0
    img_out = img_out - mean
    img_out = img_out.astype(np.float)
    img_out = im_list_to_blob([img_out])
    #The data is in this format (batch_size, channel, y, x)
    #the channel order is cv2 default: (BGR)
    return img_out


def im_list_to_blob(ims):
    #Convert a list of images into a network input.
    #Assumes images are already prepared (means subtracted, BGR order, ...).
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 4),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], 0:3] = im
        blob[i, 0:im.shape[0], 0:im.shape[1], 3] = produceCenterLabelMap(im.shape[1], im.shape[0], 21)

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def produceCenterLabelMap(wid, ht, sigma):
	X = []
	Y = []
	for j in range(ht):
		X.append([i for i in range(wid)])
		Y.append([j for i in range(wid)])
	X = np.array(X)
	X = X - wid/2.0
	Y = np.array(Y)
	Y = Y - ht/2.0
	D2 = X*X + Y*Y
	Exponent = D2 / 2.0 / sigma / sigma
	label_map = np.exp(-1 * Exponent)
	return label_map


''' -----------------Process images in various ways--------------------------'''
def process_img_scales_and_flips(net, img, norm_size, scales, heatmap_layer_name):
    heatmaps_from_multi_res = []
    for scale in scales:
        heatmaps_crop, heatmaps_flip = process_img_scale_and_flip(net, img, norm_size, scale, heatmap_layer_name)
        heatmaps_from_multi_res.append(heatmaps_crop)
        heatmaps_from_multi_res.append(heatmaps_flip)
        heatmaps_crop = []
        heatmaps_flip = []
    return heatmaps_from_multi_res


def process_img_scale_and_flip(net, img, norm_size, scale, heatmap_layer_name):
    # # 1. crop and resize image
    # img_cropped = crop_image(img, scale)
    # heatmaps_cropped = img_to_heatmaps(net, img_cropped, norm_size)
    # # 2. unzoom heatmaps (resize + padding) to be same resolution as org img
    # heatmaps_output_cropped = pad_heatmaps(heatmaps_cropped, heatmap_size, scale)
    if scale <= 1 and scale > 0:
        # 1. crop and resize image
        img_cropped = crop_image(img, scale)
        img_scaled = img_cropped
        heatmaps_cropped = img_to_heatmaps(net, img_cropped, norm_size, heatmap_layer_name)
        # 2. unzoom heatmaps (resize + padding) to be same resolution as org img
        heatmaps_output_scaled = pad_heatmaps(heatmaps_cropped, heatmap_size, scale)
    elif scale > 1:
        # 1. resize image to smaller scale and put resized image to the middle
        #    of a black image with original image size
        img_overlay = overlay_img(img, scale)
        img_scaled = img_overlay
        heatmaps_overlay = img_to_heatmaps(net, img_overlay, norm_size, heatmap_layer_name)
        # 3. recover the heatmaps to original: get small middle maps and resize
        ratio = 1/scale
        heatmaps_output_scaled = get_central_heatmaps(heatmaps_overlay, ratio, heatmap_size)

    heatmaps_flipped = process_img_flip(net, img_scaled, norm_size, heatmap_layer_name)

    if scale <= 1 and scale > 0:
        # 2. unzoom heatmaps (resize + padding) to be same resolution as org img
        heatmaps_output_flipped = pad_heatmaps(heatmaps_flipped, heatmap_size, scale)
    elif scale > 1:
        heatmaps_output_flipped = get_central_heatmaps(heatmaps_flipped, ratio, heatmap_size)

    return heatmaps_output_scaled, heatmaps_output_flipped


def process_img_scales(net, img, norm_size, scales, heatmap_layer_name):
    heatmaps_from_multi_res = []
    for scale in scales:
        heatmaps_crop = process_img_scale(net, img, norm_size, scale)
        heatmaps_from_multi_res.append(heatmaps_crop)
        heatmaps_crop = []
    return heatmaps_from_multi_res


def process_img_scale(net, img, norm_size, scale, heatmap_layer_name):
    if scale <= 1 and scale > 0:
        # 1. crop and resize image
        img_cropped = crop_image(img, scale)
        heatmaps_cropped = img_to_heatmaps(net, img_cropped, norm_size, heatmap_layer_name)
        # 2. unzoom heatmaps (resize + padding) to be same resolution as org img
        heatmaps_output = pad_heatmaps(heatmaps_cropped, heatmap_size, scale)
    elif scale > 1:
        # 1. resize image to smaller scale and put resized image to the middle
        #    of a black image with original image size
        img_overlay = overlay_img(img, scale)
        heatmaps_overlay = img_to_heatmaps(net, img_overlay, norm_size, heatmap_layer_name)
        # 3. recover the heatmaps to original: get small middle maps and resize
        ratio = 1/scale
        heatmaps_output = get_central_heatmaps(heatmaps_overlay, ratio, heatmap_size)

    return heatmaps_output


def process_img_flip(net, img, norm_size, heatmap_layer_name):
    # 1. flip image
    img_flip = cv2.flip(img, 1)
    heatmaps_flip_img = img_to_heatmaps(net, img_flip, norm_size, heatmap_layer_name)
    # 2. flip heatmaps
    heatmaps_flip_img_flipped = []
    for heatmap_flip_img in heatmaps_flip_img:
        heatmap_flip_img_flipped = cv2.flip(heatmap_flip_img, 1)
        heatmaps_flip_img_flipped.append(heatmap_flip_img_flipped)
    # 3. flip left-right order
    heatmaps_output = flip_heatmaps_order(heatmaps_flip_img_flipped)
    return heatmaps_output


def img_to_heatmaps(net, img_raw, norm_size, heatmap_layer_name):
    # pre-process image
    img = cv2.resize(img_raw, (norm_size, norm_size))
    img = preprocess(img, 0.5)

    # forwarding the Network
    output = applyDNN(img, net)

    # extract heatmaps
    heatmaps_temp = extract_heatmaps(output, heatmap_layer_name)
    return heatmaps_temp


def applyDNN(images, net):
	net_out = net.forward(data=images.astype(np.float32, copy=False))
	return net_out


def extract_heatmaps(network_output, heatmap_layer_name):
    heatmaps = []
    for key, value in network_output.items() :
        if key != heatmap_layer_name: continue
        for ith_map in range(15):
            heatmap = value[0, ith_map, :, :]
            heatmaps.append(heatmap)
    return heatmaps


''' --------------------------Demo Detections--------------------------------'''
def demo_heatmaps(heatmaps, joint_names):
    for ith_map, heatmap in enumerate(heatmaps):
        draw_heatmap(heatmap, joint_names[ith_map])


def draw_heatmap(heatmap, joint_name):
    fig = plt.figure(1, figsize=(5, 5))
    ax2 = fig.add_subplot(111)

    permut = [len(heatmap) - i- 1 for i in range(len(heatmap))]
    heatmap_flipped = heatmap[permut, :]

    ax2.imshow(heatmap_flipped, origin='lower', aspect='auto')
    ax2.set_title(joint_name)
    plt.show()
    fig.savefig('foo.png')


def demo_poses_in_img(img, joints_output, joint_pairs, joint_names):
    scale = 4
    img_demo = cv2.resize(img, None, fx= scale, fy= scale, interpolation = cv2.INTER_CUBIC)

    joints = copy.deepcopy(joints_output)
    for joint in joints:
        joint[0] *= scale
        joint[1] *= scale

    img_demo = add_joints_to_image(img_demo, joints)
    img_demo = add_joint_connections_to_image(img_demo, joints, joint_pairs, joint_names)

    if flag_demo_poses is True:
        cv2.imshow("pose image", img_demo)
        cv2.waitKey(1)
    return img_demo


def add_joints_to_image(img_demo, joints):
    for joint in joints:
        [j, i, sure] = joint
        cv2.circle(img_demo, (i, j), 8, (255,255,255), thickness=2)
    return img_demo


def add_joint_connections_to_image(img_demo, joints, joint_pairs, joint_names):
    for joint_pair in joint_pairs:
        ind_1 = joint_names.index(joint_pair[0])
        ind_2 = joint_names.index(joint_pair[1])
        if flag_color_sticks is True:
            color = find_color_scalar(joint_pair[2])
        else:
            color = find_color_scalar('red')

        y1, x1, sure1 = joints[ind_1]
        y2, x2, sure2 = joints[ind_2]
        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 ==1 and sure2 == 1:
            cv2.line(img_demo, (x1, y1), (x2, y2), color, 8)
    return img_demo


def find_color_scalar(color_string):
    color_dict = {
        'purple': (255, 0, 255),
        'yellow': (0, 255, 255),
        'blue':   (255, 0, 0),
        'green':  (0, 255, 0),
        'red':    (0, 0, 255),
        'skyblue':(235,206,135)
    }
    color_scalar = color_dict[color_string]
    return color_scalar


''' --------------------Derive joints from heatmaps -------------------------'''
def find_joints_in_heatmaps(heatmaps, thresh = 0.01):
    joints = []
    for ct, heatmap in enumerate(heatmaps):
        if ct != 14:
            joint = find_joint_in_heatmap(heatmap, thresh)
            joints.append(joint)
    return joints


def find_joint_in_heatmap(heatmap, thresh):
    peak = np.unravel_index(heatmap.argmax(), heatmap.shape)
    [j, i] = coord_by_scale(peak)

    max_val = heatmap.max()
    if max_val > thresh:
        sure = 1
    else:
        sure = 0

    joint = [j, i, sure]
    return joint


def coord_by_scale(peak):
    [j, i] = peak
    j = int(math.ceil(j * img_size / heatmap_size))
    i = int(math.ceil(i * img_size / heatmap_size))
    return [j, i]



''' ------------------Post-processing for heatmaps and images----------------'''
def flip_heatmaps_order(heatmaps):
    heatmaps_flip = []
    left_right_pairs = [ [6, 3],  # shoulder
                         [7, 4],  # elbow
                         [8, 5],  # wrist
                         [12, 9],  # pelvis
                         [13, 10],  # knee
                         [14, 11] ] # ankle
    new_order = [1, 2, 6, 7, 8, 3, 4, 5, 12, 13, 14, 9, 10, 11, 15]
    for heatmap_id in range(15):
        new_id = new_order[heatmap_id]
        heatmaps_flip.append(heatmaps[new_id - 1])
    return heatmaps_flip


def crop_image(img_raw, scale):
    scale_st = (1 - scale) / 2.0
    scale_ed = scale_st + scale
    ht, wid, channels = img_raw.shape
    pixel_st_y = ht * scale_st
    pixel_ed_y = ht * scale_ed
    pixel_st_x = wid * scale_st
    pixel_ed_x = wid * scale_ed
    img_cropped = img_raw[pixel_st_y:pixel_ed_y, pixel_st_x:pixel_ed_x]
    return img_cropped


def overlay_img(img_raw, scale):
    assert(scale > 1)
    # 1. resize image to smaller scale
    img_resized = cv2.resize(img_raw, (0, 0), fx = 1.0/scale, fy =1.0/scale)
    # 2. put resized image to the middle of a gray image with original image size
    ht, wid, channels = img_raw.shape
    ht_resized, wid_resized, channels = img_resized.shape
    x_offset = int((wid - wid_resized)/2.0)
    y_offset = int((ht - ht_resized)/2.0)

    img_overlay = img_raw.copy()
    img_overlay[:,:,:] = 128
    img_overlay[y_offset:(y_offset + ht_resized), x_offset:(x_offset + wid_resized)] = img_resized
    return img_overlay


''' ------------------------Save predictions---------------------------------'''
def save_pose_preditions(joints, save_folder, image_name, rect_id = 0):
    # joints: [[joint_1] [joint_2] [joint_3] [joint_4]]
    # the joint_id is implicitly reflected by the index in the list
    x = np.arange(15)

    base_name = os.path.splitext(image_name)[0]
    image_save_name = base_name + '.mat'
    save_path = os.path.join(save_folder, image_save_name)

    scipy.io.savemat(save_path, dict(jointids = x, joints = joints,image_name = image_name,rect_id= rect_id))
    return


def save_heatmap_preditions(heatmaps, save_folder, image_name, rect_id = 0):
    # joints: [[joint_1] [joint_2] [joint_3] [joint_4]]
    # the joint_id is implicitly reflected by the index in the list
    x = np.arange(15)

    base_name = os.path.splitext(image_name)[0]
    image_save_name = 'heatmap_' + base_name + '.mat'
    save_path = os.path.join(save_folder, image_save_name)

    scipy.io.savemat(save_path, dict(jointids = x, heatmaps = heatmaps, image_name = image_name, rect_id= rect_id))
    return


def find_rect_id(file_name):
    # Assuming input file is from MPII dataset, and file name is output by
    # [gen_cropped_test_images_with_rectid.m]
    base_name, extension = os.path.splitext(file_name)
    rect_id = int(base_name.split("_")[-1])
    return rect_id
