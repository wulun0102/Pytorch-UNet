import argparse
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import config

import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from eval import eval_net
from unet import UNet

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
from model.deeplab import Res_Deeplab
from model.pretrained_deeplab import DeepLabv3_plus
from model.pretrained_deeplab_multi import DeepLabv3_plus_multi
from model.pretrained_deeplab_multi_depth import DeepLabv3_plus_multi_depth

import wandb
# wandb.init()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#######################################
#######################################

def train_net(net,
              upsample,
              device,
              epochs=8,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):

    dataset = BasicDataset(imgs_dir=config.RGB_DIR_PATH, masks_dir=config.MASKS_DIR_PATH, depth_dir=config.DEPTH_DIR_PATH,
                           scale=img_scale, apply_imgaug=config.APPLY_IMAGE_AUG,
                           take_center_crop=config.TAKE_CENTER_CROP, crop_h=config.CROP_H, crop_w=config.CROP_W,
                           mask_suffix=config.GT_MASK_SUFFIX, depth_suffix=config.DEPTH_SUFFIX)

    if config.TRAIN_ON_SUBSET:
        #################
        ### split files
        #################
        np.random.seed(0)
        total_idx = np.arange(0, len(dataset), 1)
        train_idx = np.random.choice(total_idx, size=config.NUM_TRAIN, replace=False)
        val_idx = np.random.choice(np.delete(total_idx, train_idx), size=int(config.NUM_TRAIN*val_percent), replace=False)
        n_train, n_val = len(train_idx), len(val_idx)
        train, val = Subset(dataset, train_idx), Subset(dataset, val_idx)
    else:
        # n_val = int(len(dataset) * val_percent)
        # n_train = len(dataset) - n_val
        n_val = config.NUM_VAL
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_{config.EXPERIMENT}')
    global_step = 0

    logging.info(f'''Starting training:
        Model:           {config.MODEL_SELECTION}
        Multi Pred:      {config.MULTI_PRED}
        Use Depth:       {config.USE_DEPTH_IMAGES}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Train on subset: {config.TRAIN_ON_SUBSET}
        Training size:   {n_train}
        Validation size: {n_val}
        Images scaling:  {img_scale}
        Crop images:     {config.TAKE_CENTER_CROP}
        Image Size:      {config.CROP_H}, {config.CROP_W}
        Apply imgaug:    {config.APPLY_IMAGE_AUG}
        Device:          {device.type}
    ''')

    ##################
    # SEGMENTATION
    ##################

    if config.MODEL_SELECTION == 'og_deeplab':
        optimizer = optim.RMSprop(net.optim_parameters(lr=lr), lr=lr, weight_decay=1e-8, momentum=0.9)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    if config.NUM_CLASSES > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if config.NUM_CLASSES > 1 else 'max', patience=2)

    if save_cp:
        try:
            os.mkdir(config.CHECKPOINT_DIR_PATH)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

    best_Fwb = -np.inf
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                depths = batch['depth']
                true_masks = batch['mask']
                assert imgs.shape[1] == config.NUM_CHANNELS, \
                    f'Network has been defined with {config.NUM_CHANNELS} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                depths = depths.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if config.NUM_CLASSES == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                if config.MULTI_PRED:
                ###################################
                # multi
                ###################################
                    if config.USE_DEPTH_IMAGES:
                        masks_pred1, masks_pred2 = net(imgs, depths)
                    else:
                        masks_pred1, masks_pred2 = net(imgs)
                    loss = criterion(masks_pred1, true_masks.squeeze(1)) + criterion(masks_pred2, true_masks.squeeze(1))

                else:
                ###################################
                # single
                ###################################
                    if config.MODEL_SELECTION == 'og_deeplab':
                        masks_pred = upsample(net(imgs))
                    else:
                        masks_pred = net(imgs)
                    loss = criterion(masks_pred, true_masks.squeeze(1))

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # if global_step % (n_train // (10 * batch_size)) == 0:
                if global_step % (n_train // (1 * batch_size)) == 0:
                    # segmentation model
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        if value.grad is None:
                            # print('Layer: ', tag.split('/'))
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            pass
                        else:
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    # weighted fwb score
                    val_loss, Fwb = eval_net(net, upsample, val_loader, writer, best_Fwb, global_step, device)
                    writer.add_scalar('Weighted-Fb/Current-Fwb', Fwb, global_step)

                    scheduler.step(val_loss)
                    writer.add_scalar('learning_rate/seg', optimizer.param_groups[0]['lr'], global_step)

                    if config.NUM_CLASSES > 1:
                        writer.add_scalar('Loss/test', val_loss, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(Fwb))
                        writer.add_scalar('Dice/test', Fwb, global_step)

                    if Fwb > best_Fwb and save_cp:
                        best_Fwb = Fwb
                        writer.add_scalar('Weighted-Fb/Best-Fwb', best_Fwb, global_step)
                        torch.save(net.state_dict(), config.BEST_MODEL_SAVE_PATH)
                        logging.info('Best Model Saved with Fwb: {:.5}!'.format(best_Fwb))

    if save_cp:
        torch.save(net.state_dict(), config.MODEL_SAVE_PATH + "Epoch_{}_Best_Seg_{:.5}.pth".format((config.EPOCHS, best_Fwb)))
        logging.info('Final Model Saved with Fwb: {:.5}!'.format(best_Fwb))
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=config.EPOCHS,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=config.BATCH_SIZE,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=config.LR,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0/100.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #######################
    # Segmentation
    #######################

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    if config.MODEL_SELECTION == 'og_unet':
        net = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES, bilinear=True)
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
        raise NotImplementedError

    upsample = nn.Upsample(size=(config.CROP_H, config.CROP_W), mode='bilinear', align_corners=True)

    #######################
    #######################

    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True

    from torchsummary import summary
    if config.USE_DEPTH_IMAGES:
        summary(net, [(config.NUM_CHANNELS, config.CROP_H, config.CROP_W),
                      (config.NUM_CHANNELS, config.CROP_H, config.CROP_W)])
    else:
        summary(net, (config.NUM_CHANNELS, config.CROP_H, config.CROP_W))

    logging.info(f'Network:\n'
                 f'\t{config.NUM_CHANNELS} input channels\n'
                 f'\t{config.NUM_CLASSES} output channels (classes)\n')

    if config.SEG_SAVED_WEIGHTS:
        net.load_state_dict(torch.load(config.SEG_SAVED_WEIGHTS, map_location=device))
        logging.info(f'Model loaded from {config.SEG_SAVED_WEIGHTS}!\n')
    else:
        logging.info(f'Training Model from scratch!\n')


    #######################
    #######################

    try:
        train_net(net=net,
                  upsample=upsample,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
