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

from torch.autograd import Variable
import torch.nn.functional as F

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

from model.discriminator import FCDiscriminator

import wandb
# wandb.init()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#######################################
#######################################

def train_net(net,
              discriminator1,
              discriminator2,
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
                           extend_dataset=True, num_images=int(config.ITERATIONS + config.NUM_VAL),
                           scale=img_scale, apply_imgaug=config.APPLY_IMAGE_AUG,
                           take_center_crop=config.TAKE_CENTER_CROP, crop_h=config.CROP_H, crop_w=config.CROP_W,
                           mask_suffix=config.TARGET_GT_MASK_SUFFIX)

    n_val = config.NUM_VAL
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    logging.info(f'Real - Train dataset has {n_train} examples')
    logging.info(f'Real - Val dataset has {n_val} examples')

    target_train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    target_val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    target_train_iterator = enumerate(target_train_loader)

    ##########################
    ##########################

    writer = SummaryWriter(comment=f'_ADV_{config.EXPERIMENT}')
    global_step = 0

    logging.info(f'''Starting training:
        Model:           {config.MODEL_SELECTION}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Training size:   {n_train}
        Val size:        {n_val}
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

    ##################
    # DISCRIMINATOR
    ##################
    optimizer_discriminator1 = optim.Adam(discriminator1.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer_discriminator2 = optim.Adam(discriminator2.parameters(), lr=lr, betas=(0.9, 0.99))

    if config.GAN == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif config.GAN == 'LS':
        bce_loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    ##################
    # TODO: LR scheduler
    ##################

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if config.NUM_CLASSES > 1 else 'max', patience=2)
    # scheduler_discriminator1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator1, 'min' if config.NUM_CLASSES > 1 else 'max', patience=2)
    # scheduler_discriminator2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator1, 'min' if config.NUM_CLASSES > 1 else 'max', patience=2)

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate(optimizer, i_iter, lr=config.LR):
        lr = lr_poly(lr, i_iter, config.ITERATIONS, config.POWER)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def adjust_learning_rate_D(optimizer, i_iter, lr=config.LR):
        lr = lr_poly(lr, i_iter, config.ITERATIONS, config.POWER)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    ##################
    ##################

    net.train()
    discriminator1.train()
    discriminator2.train()
    best_Fwb = -np.inf

    # labels for adversarial training
    target_label = 0
    source_label = 1

    if save_cp:
        try:
            os.mkdir(config.CHECKPOINT_DIR_PATH)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

    with tqdm(total=config.ITERATIONS, desc=f'Iterations {config.ITERATIONS}', unit='img') as pbar:
        while global_step < config.ITERATIONS:

            seg_loss = 0
            seg_adv_loss = 0
            dis_loss1 = 0
            dis_loss2 = 0

            optimizer.zero_grad()
            optimizer_discriminator1.zero_grad()
            optimizer_discriminator2.zero_grad()

            adjust_learning_rate(optimizer, global_step)
            adjust_learning_rate_D(optimizer_discriminator1, global_step)
            adjust_learning_rate_D(optimizer_discriminator2, global_step)

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

            if config.MULTI_PRED:
                ###################################
                # multi
                ###################################
                masks_pred1_source, masks_pred2_source = net(imgs)
                loss = criterion(masks_pred1_source, true_masks.squeeze(1)) + \
                       config.LAMBDA_SEG * criterion(masks_pred2_source, true_masks.squeeze(1))

            seg_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)

            writer.add_scalar('Loss/train', seg_loss, global_step)

            ##########################
            # Seg w./ Target
            ##########################
            # don't accumulate grads in D
            for param in discriminator1.parameters():
                param.requires_grad = False

            for param in discriminator2.parameters():
                param.requires_grad = False

            ###############
            ###############
            _, batch = target_train_iterator.__next__()
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=torch.float32)
            # true_masks = batch['mask']
            # true_masks = true_masks.to(device=device, dtype=mask_type)

            if config.MULTI_PRED:
                ###################################
                # multi
                ###################################
                masks_pred1_target, masks_pred2_target = net(imgs)

            discriminator_out1_target = discriminator1(F.softmax(masks_pred1_target))
            discriminator_out2_target = discriminator2(F.softmax(masks_pred2_target))
            discriminator_adv1_fill = Variable(
                torch.FloatTensor(discriminator_out1_target.data.size()).fill_(source_label)).to(device=device)
            discriminator_adv2_fill = Variable(
                torch.FloatTensor(discriminator_out2_target.data.size()).fill_(source_label)).to(device=device)

            loss = config.ADV_SEG1 * bce_loss(discriminator_out1_target, discriminator_adv1_fill) + \
                   config.ADV_SEG2 * bce_loss(discriminator_out2_target, discriminator_adv2_fill)
            seg_adv_loss += loss.item()
            loss.backward()

            writer.add_scalar('Adv_Loss/Adv', seg_adv_loss, global_step)

            #############################
            # DISCRIMINATOR w/ Target
            #############################

            # now accumulate grads in D
            for param in discriminator1.parameters():
                param.requires_grad = True

            for param in discriminator2.parameters():
                param.requires_grad = True

            masks_pred1_target = masks_pred1_target.detach()
            masks_pred2_target = masks_pred2_target.detach()

            discriminator_out1_target = discriminator1(F.softmax(masks_pred1_target))
            discriminator_out2_target = discriminator2(F.softmax(masks_pred2_target))
            discriminator_fill1_target = Variable(
                torch.FloatTensor(discriminator_out1_target.data.size()).fill_(target_label)).to(device=device)
            discriminator_fill2_target = Variable(
                torch.FloatTensor(discriminator_out2_target.data.size()).fill_(target_label)).to(device=device)

            loss_D1 = bce_loss(discriminator_out1_target, discriminator_fill1_target) / 2
            loss_D2 = bce_loss(discriminator_out2_target, discriminator_fill2_target) / 2
            dis_loss1 += loss_D1.item()
            dis_loss2 += loss_D2.item()
            loss_D1.backward()
            loss_D2.backward()

            #############################
            # DISCRIMINATOR w/ Source
            #############################
            masks_pred1_source = masks_pred1_source.detach()
            masks_pred2_source = masks_pred2_source.detach()

            discriminator_out1_source = discriminator1(F.softmax(masks_pred1_source))
            discriminator_out2_source = discriminator2(F.softmax(masks_pred2_source))
            discriminator_fill1_source = Variable(
                torch.FloatTensor(discriminator_out1_source.data.size()).fill_(source_label)).to(device=device)
            discriminator_fill2_source = Variable(
                torch.FloatTensor(discriminator_out2_source.data.size()).fill_(source_label)).to(device=device)

            loss_D1 = bce_loss(discriminator_out1_source, discriminator_fill1_source) / 2
            loss_D2 = bce_loss(discriminator_out2_source, discriminator_fill2_source) / 2
            dis_loss1 += loss_D1.item()
            dis_loss2 += loss_D2.item()
            loss_D1.backward()
            loss_D2.backward()

            writer.add_scalar('Adv_Loss/Discriminator_1', dis_loss1, global_step)
            writer.add_scalar('Adv_Loss/Discriminator_2', dis_loss2, global_step)

            ##########################
            ##########################

            optimizer.step()
            optimizer_discriminator1.step()
            optimizer_discriminator2.step()

            pbar.set_postfix(**{'loss ': seg_loss,
                                'adv_loss ': seg_adv_loss,
                                'dis_loss1 ': dis_loss1,
                                'dis_loss2 ': dis_loss2})

            global_step += 1
            pbar.update(imgs.shape[0])

            global_step += 1
            if global_step % (config.NUM_IMAGES_PER_EPOCH // (1 * batch_size)) == 0:
                # segmentation
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        print('Seg_Layer: ', tag.split('/'))
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        pass
                    else:
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                # discriminator
                for tag, value in discriminator1.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        print('Dis1_Layer: ', tag.split('/'))
                        writer.add_histogram('dis_weights1/' + tag, value.data.cpu().numpy(), global_step)
                        pass
                    else:
                        writer.add_histogram('dis_weights1/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('dis_grads1/' + tag, value.grad.data.cpu().numpy(), global_step)

                # discriminator
                for tag, value in discriminator2.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        print('Dis2_Layer: ', tag.split('/'))
                        writer.add_histogram('dis_weights2/' + tag, value.data.cpu().numpy(), global_step)
                        pass
                    else:
                        writer.add_histogram('dis_weights2/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('dis_grads2/' + tag, value.grad.data.cpu().numpy(), global_step)

                # weighted fwb score
                val_loss, Fwb = eval_net(net, upsample, target_val_loader, writer, global_step, device)
                writer.add_scalar('Weighted-Fb/Current-Fwb', Fwb, global_step)


                # scheduler.step(val_loss)
                # scheduler_discriminator1.step(val_loss)
                # scheduler_discriminator2.step(val_loss)
                writer.add_scalar('learning_rate/seg', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('learning_rate/dis1', optimizer_discriminator1.param_groups[0]['lr'], global_step)
                writer.add_scalar('learning_rate/dis2', optimizer_discriminator2.param_groups[0]['lr'], global_step)

                if config.NUM_CLASSES > 1:
                    writer.add_scalar('Loss/test', val_loss, global_step)
                else:
                    logging.info('Validation Dice Coeff: {}'.format(Fwb))
                    writer.add_scalar('Dice/test', Fwb, global_step)

                if Fwb > best_Fwb and save_cp:
                    best_Fwb = Fwb
                    writer.add_scalar('Weighted-Fb/Best-Fwb', best_Fwb, global_step)
                    torch.save(net.state_dict(), config.BEST_MODEL_SAVE_PATH)
                    torch.save(discriminator1.state_dict(), config.BEST_DIS1_SAVE_PATH)
                    torch.save(discriminator2.state_dict(), config.BEST_DIS2_SAVE_PATH)
                    logging.info('Best Model Saved with Fwb: {:.5}!'.format(best_Fwb))

    if save_cp:
        torch.save(net.state_dict(), config.MODEL_SAVE_PATH + "Best_Seg_{:.5}.pth".format((best_Fwb)))
        torch.save(discriminator1.state_dict(), config.MODEL_SAVE_PATH + "Best_Dis1_{:.5}.pth".format((best_Fwb)))
        torch.save(discriminator2.state_dict(), config.MODEL_SAVE_PATH + "Best_Dis2_{:.5}.pth".format((best_Fwb)))
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
    elif config.MODEL_SELECTION == 'og_deeplab':
        net = Res_Deeplab(num_classes=config.NUM_CLASSES)
    else:
        raise NotImplementedError

    upsample = nn.Upsample(size=(config.CROP_H, config.CROP_W), mode='bilinear', align_corners=True)

    #######################
    #######################

    logging.info(f'Network:\n'
                 f'\t{config.NUM_CHANNELS} input channels\n'
                 f'\t{config.NUM_CLASSES} output channels (classes)\n')

     # TODO:
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True

    from torchsummary import summary
    summary(net, (config.NUM_CHANNELS, config.CROP_H, config.CROP_W))

    #######################
    # Discriminator
    #######################

    discriminator1 = FCDiscriminator(num_classes=config.NUM_CLASSES)
    discriminator1.to(device=device)

    discriminator2 = FCDiscriminator(num_classes=config.NUM_CLASSES)
    discriminator2.to(device=device)

    logging.info(f'Discriminator:\n'
                 f'\t{config.NUM_CLASSES} input channels (classes)\n')

    from torchsummary import summary
    summary(discriminator1, (config.NUM_CLASSES, config.CROP_H, config.CROP_W))
    summary(discriminator2, (config.NUM_CLASSES, config.CROP_H, config.CROP_W))

    #######################
    #######################

    try:
        train_net(net=net,
                  discriminator1=discriminator1,
                  discriminator2=discriminator2,
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
