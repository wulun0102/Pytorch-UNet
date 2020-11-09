import argparse
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import config
import time

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

from torch.autograd import Variable
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from model.og_deeplab import Res_Deeplab
from model.deeplab import DeepLabv3_plus
from model.discriminator import FCDiscriminator

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#######################################
#######################################

def train_net(net,
              discriminator,
              upsample,
              device,
              epochs=8,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):

    ##########################
    # creating syn dataset
    ##########################
    dataset = BasicDataset(config.SOURCE_RGB_DIR_PATH, config.SOURCE_MASKS_DIR_PATH,
                           extend_dataset=True, num_images=config.ITERATIONS,
                           scale=img_scale, apply_imgaug=config.APPLY_IMAGE_AUG,
                           take_center_crop=config.TAKE_CENTER_CROP, crop_h=config.CROP_H, crop_w=config.CROP_W,
                           mask_suffix=config.SOURCE_GT_MASK_SUFFIX)

    source_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    source_train_iterator = enumerate(source_train_loader)

    ##########################
    # creating real dataset
    ##########################
    dataset = BasicDataset(config.TARGET_RGB_DIR_PATH, config.TARGET_MASKS_DIR_PATH,
                           extend_dataset=True, num_images=config.ITERATIONS,
                           scale=img_scale, apply_imgaug=config.APPLY_IMAGE_AUG,
                           take_center_crop=config.TAKE_CENTER_CROP, crop_h=config.CROP_H, crop_w=config.CROP_W,
                           mask_suffix=config.TARGET_GT_MASK_SUFFIX)

    target_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    target_train_iterator = enumerate(source_train_loader)

    ##########################
    ##########################

    writer = SummaryWriter(comment=f'_ADV_LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Model:           {config.MODEL_SELECTION}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Train on subset: {config.TRAIN_ON_SUBSET}
        Training size:   {config.NUM_IMAGES_PER_EPOCH}
        Images scaling:  {img_scale}
        Crop images:     {config.TAKE_CENTER_CROP}
        Image Size:      {config.CROP_H}, {config.CROP_W}
        Apply imgaug:    {config.APPLY_IMAGE_AUG}
        Device:          {device.type}
    ''')

    ##################
    # SEGMENTATION
    ##################

    if config.MODEL_SELECTION == 'adapt_deeplab':
        optimizer = optim.RMSprop(net.optim_parameters(lr=lr), lr=lr, weight_decay=1e-8, momentum=0.9)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    if config.NUM_CLASSES > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    ##################
    # DISCRIMINATOR
    ##################
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.99))

    if config.GAN == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif config.GAN == 'LS':
        bce_loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    ##################
    ##################

    # TODO LR scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if config.NUM_CLASSES > 1 else 'max', patience=2)

    net.train()
    discriminator.train()

    # labels for adversarial training
    target_label = 0
    source_label = 1

    with tqdm(total=config.ITERATIONS, desc=f'Iterations {config.ITERATIONS}', unit='img') as pbar:
        for iteration in range(config.ITERATIONS):

            iteration_loss = 0
            iteration_adv_loss = 0
            iteration_discriminator_loss = 0

            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()

            ##########################
            # Seg w./ Source
            ##########################
            _, batch = source_train_iterator.__next__()
            imgs = batch['image']
            true_masks = batch['mask']

            assert imgs.shape[1] == config.NUM_CHANNELS, \
                f'Network has been defined with {config.NUM_CHANNELS} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if config.NUM_CLASSES == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if config.MODEL_SELECTION == 'adapt_deeplab':
                masks_pred_source = upsample(net(imgs))
            else:
                masks_pred_source = net(imgs)

            loss = criterion(masks_pred_source, true_masks.squeeze(1))
            iteration_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            ##########################
            # Seg w./ Target
            ##########################
            # don't accumulate grads in D
            for param in discriminator.parameters():
                param.requires_grad = False

            ###
            _, batch = target_train_iterator.__next__()
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=torch.float32)
            # true_masks = batch['mask']
            # true_masks = true_masks.to(device=device, dtype=mask_type)

            if config.MODEL_SELECTION == 'adapt_deeplab':
                masks_pred_target = upsample(net(imgs))
            else:
                masks_pred_target = net(imgs)

            discriminator_out_target = discriminator(F.softmax(masks_pred_target))
            discriminator_adv_fill = Variable(
                torch.FloatTensor(discriminator_out_target.data.size()).fill_(source_label)).to(device=device)

            loss = bce_loss(discriminator_out_target, discriminator_adv_fill)
            iteration_adv_loss += loss.item()
            loss.backward()

            writer.add_scalar('Adv_Loss/train', loss.item(), global_step)

            #############################
            # DISCRIMINATOR w/ Target
            #############################

            # now accumulate grads in D
            for param in discriminator.parameters():
                param.requires_grad = True

            masks_pred_target = masks_pred_target.detach()

            discriminator_out_target = discriminator(F.softmax(masks_pred_target))
            discriminator_fill_target = Variable(
                torch.FloatTensor(discriminator_out_target.data.size()).fill_(target_label)).to(device=device)

            loss = bce_loss(discriminator_out_target, discriminator_fill_target) / 2 # TODO: ??
            iteration_discriminator_loss += loss.item()
            loss.backward()

            #############################
            # DISCRIMINATOR w/ Source
            #############################
            masks_pred_source = masks_pred_source.detach()

            discriminator_out_source = discriminator(F.softmax(masks_pred_source))
            discriminator_fill_source = Variable(
                torch.FloatTensor(discriminator_out_source.data.size()).fill_(source_label)).to(device=device)

            loss = bce_loss(discriminator_out_source, discriminator_fill_source) / 2 # TODO: ??
            iteration_discriminator_loss += loss.item()
            loss.backward()

            writer.add_scalar('Discriminator_Loss/train', iteration_discriminator_loss, global_step)

            ##########################
            ##########################

            optimizer.step()
            optimizer_discriminator.step()

            pbar.update(imgs.shape[0])
            pbar.set_postfix(**{'loss ': iteration_loss,
                                'adv_loss ': iteration_adv_loss,
                                'dis_loss ': iteration_discriminator_loss})

            global_step += 1

            if global_step % 2000 == 0:
                # segmentation
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        print('Layer: ', tag.split('/'))
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        pass
                    else:
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                # discriminator
                for tag, value in discriminator.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        print('Layer: ', tag.split('/'))
                        writer.add_histogram('dis_weights/' + tag, value.data.cpu().numpy(), global_step)
                        pass
                    else:
                        writer.add_histogram('dis_weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('dis_grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                # scheduler.step(iteration_loss / global_step)
                # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                # writer.add_images('images', imgs, global_step)

                if save_cp:
                    try:
                        os.mkdir(config.CHECKPOINT_DIR_PATH)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    # torch.save(net.state_dict(),config.CHECKPOINT_DIR_PATH + f'CP_{global_step}.pth')
                    # torch.save(discriminator.state_dict(),config.CHECKPOINT_DIR_PATH + f'CP_{global_step}_DIS.pth')
                    torch.save(net.state_dict(), config.CHECKPOINT_DIR_PATH + f'NET.pth')
                    torch.save(discriminator.state_dict(), config.CHECKPOINT_DIR_PATH + f'DIS.pth')
                    logging.info(f'Checkpoint {global_step} saved !')

    ##########################
    ##########################

    # saving final model
    torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
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
    parser.add_argument('-v', '--validation', dest='val', type=float, default=5.0/100.0,
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
    elif config.MODEL_SELECTION == 'pytorch_deeplab':
        net = DeepLabv3_plus(nInputChannels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES,
                             os=16, pretrained=True, _print=False)
    elif config.MODEL_SELECTION == 'adapt_deeplab':
        net = Res_Deeplab(num_classes=config.NUM_CLASSES)
    else:
        raise NotImplementedError

    upsample = nn.Upsample(size=(config.CROP_H, config.CROP_W), mode='bilinear', align_corners=True)

    #######################
    #######################

    logging.info(f'Network:\n'
                 f'\t{config.NUM_CHANNELS} input channels\n'
                 f'\t{config.NUM_CLASSES} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    from torchsummary import summary
    summary(net, (config.NUM_CHANNELS, config.CROP_H, config.CROP_W))

    #######################
    # Discriminator
    #######################

    discriminator = FCDiscriminator(num_classes=config.NUM_CLASSES)
    discriminator.to(device=device)

    logging.info(f'Discriminator:\n'
                 f'\t{config.NUM_CLASSES} input channels (classes)\n')

    from torchsummary import summary
    summary(discriminator, (config.NUM_CLASSES, config.CROP_H, config.CROP_W))

    #######################
    #######################

    try:
        train_net(net=net,
                  discriminator=discriminator,
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
