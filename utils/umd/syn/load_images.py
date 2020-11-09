import cv2
import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import argparse

import skimage.io

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Load YCB Images')

parser.add_argument('--dataset', required=False,
                    default='/data/Akeaveny/Datasets/part-affordance_combined/UNET/syn/umd/objects/hammer/test/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

args = parser.parse_args()

#########################
# load images
#########################

rgb_folder = 'rgb/'
rgb_ext = ".png"

mask_folder = 'masks/'
mask_ext = "_gt_affordance.png"

depth_folder = 'depth/'
depth_ext = "_depth.png"

image_path_ = args.dataset + depth_folder + "*" + depth_ext
image_files = sorted(glob.glob(image_path_))
image_max_depth = -np.inf
image_min_depth = np.inf
NDDS_DEPTH_CONST = 10e3 / (2 ** 8 - 1)

print('Loaded {} Images'.format(len(image_files)))

for idx, depth_image_addr in enumerate(image_files):

    #####################
    # See 3 diff depths
    #####################
    image_idx = depth_image_addr.split(args.dataset + depth_folder)[1].split(depth_ext)[0]

    rgb_addr = args.dataset + rgb_folder + str(image_idx) + rgb_ext
    mask_addr = args.dataset + mask_folder + str(image_idx) + mask_ext

    ### load images
    depth = np.array(Image.open(depth_image_addr)) * NDDS_DEPTH_CONST
    rgb = np.array(Image.open(rgb_addr))
    mask = np.array(Image.open(mask_addr))

    ### plot
    plt.figure(0)
    plt.subplot(1, 3, 1)
    plt.title("depth")
    plt.imshow(depth)
    plt.subplot(1, 3, 2)
    plt.title("rgb")
    plt.imshow(rgb)
    plt.subplot(1, 3, 3)
    plt.title("mask")
    plt.imshow(mask)
    plt.show()
    plt.ioff()

    #####################
    # MAX DEPTH
    #####################
    # depth = np.array(Image.open(depth_image_addr)) * NDDS_DEPTH_CONST
    #
    # max_depth = np.max(depth)
    # min_depth = np.min(depth)
    #
    # image_max_depth = max_depth if max_depth > image_max_depth else image_max_depth
    # image_min_depth = min_depth if min_depth > image_min_depth else image_min_depth

print("Img dtype:\t{}, Min Depth:\t{}, Max Depth:\t{}".format(depth.dtype, image_min_depth, image_max_depth))







