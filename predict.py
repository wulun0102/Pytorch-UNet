import argparse
import os
import glob

import config

import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import skimage.transform

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset

import segmentation_models_pytorch as smp
from model.deeplab import Res_Deeplab
from model.pretrained_deeplab import DeepLabv3_plus
from model.pretrained_deeplab_multi import DeepLabv3_plus_multi
from model.pretrained_deeplab_multi_depth import DeepLabv3_plus_multi_depth

from weighted_Fb import weightedFb

####################################
#
####################################

def model_ouput_to_pred_mask(args, model, upsample, image, depth, gt, scale_factor=1):

    rgb = torch.from_numpy(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    depths = torch.from_numpy(depth).unsqueeze(0).to(device=device, dtype=torch.float32)
    # print("rgb size: ", rgb.shape)

    #######################
    # get pred from model
    #######################

    with torch.no_grad():
        if config.MULTI_PRED:
            ###################################
            # multi
            ###################################
            if config.USE_DEPTH_IMAGES:
                output1, output2 = net(rgb, depths)
            else:
                output1, output2 = net(rgb)
            output = output1

        else:
            ###################################
            # single
            ###################################
            if config.MODEL_SELECTION == 'og_deeplab':
                output = upsample(net(rgb))
            else:
                output = net(rgb)

    if config.NUM_CLASSES > 1:
        probs = F.softmax(output, dim=1)
    else:
        probs = torch.sigmoid(output)

    pred_mask = probs.squeeze(0).cpu().numpy()
    final_mask = np.squeeze(np.asarray(np.argmax(pred_mask, axis=0), dtype=np.uint8))

    #######################
    #######################

    # aff_ids, rows, cols = pred_mask.shape
    # final_mask = np.zeros(shape=(rows,cols))
    # for row in range(rows):
    #     for col in range(cols):
    #         # TODO: Confidence - 1/args.num_classes
    #         # print("Max: ", np.max(pred_mask[:, row, col]))
    #         aff_id = np.argmax(pred_mask[:, row, col])
    #         final_mask[row, col] = aff_id if pred_mask[aff_id, row, col] > 1/config.NUM_CLASSES else 0 # background
    # final_mask = np.array(final_mask, dtype=np.uint8)

    #######################
    #######################

    # print("final_mask.shape: ", final_mask.shape, 'gt.shape: ', gt.squeeze(0).size())
    # assert final_mask.shape == image.squeeze(0).size()

    if args.visualize:
        ### plotting
        plt.figure(0)
        plt.subplot(2, 2, 1)
        plt.title("RGB")
        plt.imshow(image.transpose(1, 2, 0))
        plt.subplot(2, 2, 2)
        plt.title("DEPTH")
        plt.imshow(depth.transpose(1, 2, 0))
        plt.subplot(2, 2, 3)
        plt.title("gt")
        plt.imshow(gt)
        print("gt: ", np.unique(gt))
        plt.subplot(2, 2, 4)
        plt.title("pred")
        plt.imshow(final_mask)
        print("pred_mask: ", np.unique(final_mask))
        plt.show()
        plt.ioff()

    return final_mask

####################################
#
####################################

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m',
                        default=config.MODEL_SAVE_PATH,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--dataset', '-i', metavar='INPUT', nargs='+',
                        default='',
                        help='filenames of input images')

    parser.add_argument('--mask-threshold', '-t', type=float,
                        default=0.5,
                        help="Minimum probability value to consider a mask pixel white")

    parser.add_argument('--scale', '-s', type=float,
                        default=1,
                        help="Scale factor for the input images")

    parser.add_argument('--visualize', '-v', type=bool,
                        default=False,
                        help="Scale factor for the input images")

    return parser.parse_args()

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    #######################
    # Segmentation
    #######################

    if config.MODEL_SELECTION == 'og_unet':
        net = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES)
    elif config.MODEL_SELECTION == 'smp_unet':
        net = smp.Unet(config.BACKBONE, encoder_weights=config.ENCODER_WEIGHTS, classes=config.NUM_CLASSES)
    elif config.MODEL_SELECTION == 'smp_fpn':
        net = smp.FPN(config.BACKBONE, encoder_weights=config.ENCODER_WEIGHTS, classes=config.NUM_CLASSES)
    elif config.MODEL_SELECTION == 'pretrained_deeplab':
        net = DeepLabv3_plus(nInputChannels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES,
                             os=16, pretrained=True, _print=False)
    elif config.MODEL_SELECTION == 'pretrained_deeplab_multi':
        net = DeepLabv3_plus_multi(nInputChannels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES,
                                    os=16, pretrained=True, _print=False)
    elif config.MODEL_SELECTION == 'pretrained_deeplab_multi_depth':
        net = DeepLabv3_plus_multi_depth(nInputChannels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES,
                                         os=16, pretrained=True, _print=False)
    elif config.MODEL_SELECTION == 'og_deeplab':
        net = Res_Deeplab(num_classes=config.NUM_CLASSES)
    else:
        logging.info("*** No Model Selected! ***")
        exit(0)

    upsample = nn.Upsample(size=(config.CROP_H, config.CROP_W), mode='bilinear', align_corners=True)

    #######################
    #######################

    logging.info(f'Loading model {args.model}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    logging.info("Model loaded !")

    ########################
    ########################

    image_path_ = config.TEST_RGB_DIR_PATH + "*" + config.RGB_IMG_EXT
    image_files = sorted(glob.glob(image_path_))
    logging.info(f'Loaded {len(image_files)} Images from {image_path_}')

    ###############
    # split files
    ###############
    np.random.seed(0)
    total_idx = np.arange(0, len(image_files), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_files = total_idx[test_idx]
    logging.info(f'Choosing {len(test_files)} Images')

    for idx, image_num in enumerate(test_files):

        img_idx = 1000000 + image_num
        img_idx = str(img_idx)[1:]

        logging.info(f'\tPredicting on image {idx+1} (or {img_idx}) ...')

        image_addr = config.TEST_RGB_DIR_PATH + img_idx + config.RGB_IMG_EXT
        img = Image.open(image_addr)

        depth_addr = config.TEST_DEPTH_DIR_PATH + img_idx + config.DEPTH_IMG_EXT
        depth = Image.open(depth_addr)
        depth = Image.fromarray(skimage.color.gray2rgb(np.array(depth)))

        gt_mask_addr = config.TEST_MASKS_DIR_PATH + img_idx + config.GT_MASK_EXT
        gt = Image.open(gt_mask_addr)

        # preprocess
        img = BasicDataset.crop(img, config.CROP_H, config.CROP_H, is_img=True)
        img = BasicDataset.preprocess(Image.fromarray(img), args.scale, is_img=True)
        depth = BasicDataset.crop(depth, config.CROP_H, config.CROP_H, is_img=True)
        depth = BasicDataset.preprocess(Image.fromarray(depth), args.scale, is_img=True)
        gt = BasicDataset.crop(gt, config.CROP_H, config.CROP_H, is_img=False)
        gt = Image.fromarray(gt)

        pred_gt_mask_addr = config.TEST_PRED_DIR_PATH + img_idx + config.GT_SAVE_MASK_EXT
        gt.save(pred_gt_mask_addr)

        pred_mask_addr = config.TEST_PRED_DIR_PATH + img_idx + config.PRED_MASK_EXT

        mask = model_ouput_to_pred_mask(args, net, upsample, img, depth, gt)
        pred = Image.fromarray(mask)
        pred.save(pred_mask_addr)
    # get Fwb score
    Fwb = weightedFb(config.TEST_PRED_DIR_PATH, aff_start=config.AFFORDANCE_START, aff_end=config.AFFORDANCE_END,
                     VERBOSE=True, VISUALIZE=False)

