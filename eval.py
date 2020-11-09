from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from tqdm import tqdm

import config
from weighted_Fb import weightedFb

def eval_net(net, upsample, loader, writer, global_step, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if config.NUM_CLASSES == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_loss = 0

    if config.NUM_CLASSES > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    img_idx = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            with torch.no_grad():
                if config.MODEL_SELECTION == 'adapt_deeplab':
                    mask_pred = upsample(net(imgs))
                else:
                    mask_pred = net(imgs)

            ###################
            # val loss
            ###################
            if config.NUM_CLASSES > 1:
                tot_loss += criterion(mask_pred, true_masks.squeeze(1)).item()
            else:
                raise NotImplementedError

            ###################
            # getting prediction
            ###################

            if config.NUM_CLASSES > 1:
                probs = F.softmax(mask_pred, dim=1)
            else:
                probs = torch.sigmoid(mask_pred)

            # print("gt_mask: ", true_masks.shape)
            # print("pred_mask: ", probs.shape)
            # probs are N x C X H X W
            pred_mask = probs.squeeze(0).cpu().numpy()
            final_mask = np.squeeze(np.asarray(np.argmax(pred_mask, axis=1), dtype=np.uint8))

            true_masks = true_masks.squeeze(0).cpu().numpy()
            true_masks = np.squeeze(np.asarray(true_masks, dtype=np.uint8))

            gt = true_masks[:, np.newaxis]
            pred = final_mask[:, np.newaxis]
            # print("gt_mask: ", gt.shape)
            # print("pred_mask: ", pred.shape)

            if img_idx == 0:
                writer.add_images('gt_mask', gt * 40, global_step)
                writer.add_images('pred_mask', pred * 40, global_step)

            ###################
            # saving images for Fwb
            ###################

            for i in range(config.BATCH_SIZE):
                gt_mask_addr = config.TEST_PRED_DIR_PATH + str(img_idx) + config.GT_SAVE_MASK_EXT
                gt_test = np.squeeze(gt[i, :, :, :])
                Image.fromarray(gt_test).save(gt_mask_addr)

                pred_mask_addr = config.TEST_PRED_DIR_PATH + str(img_idx) + config.PRED_MASK_EXT
                pred_test = np.squeeze(pred[i, :, :, :])
                Image.fromarray(pred_test).save(pred_mask_addr)

                # #################
                # plt.figure(0)
                # plt.subplot(1, 2, 1)
                # plt.title("gt_test")
                # plt.imshow(gt_test)
                # print("gt_test: ", gt_test.shape)
                # plt.subplot(1, 2, 2)
                # plt.title("pred_test")
                # plt.imshow(pred_test)
                # print("pred_test: ", pred_test.shape)
                # plt.show()
                # plt.ioff()
                # #################

                img_idx += 1

            pbar.update()

    val_loss = tot_loss/n_val
    Fwb = weightedFb(config.TEST_PRED_DIR_PATH, aff_start=config.AFFORDANCE_START, aff_end=config.AFFORDANCE_END,
                     VERBOSE=False, VISUALIZE=False)
    net.train()
    return val_loss, Fwb
