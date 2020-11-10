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
from model.pretrained_deeplab_multi_depth import DeepLabv3_plus_multi_depth

from model.discriminator import FCDiscriminator

from loss import CrossEntropy2d
from loss import WeightedBCEWithLogitsLoss

import wandb
# wandb.init()

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
    dataset = BasicDataset(imgs_dir=config.SOURCE_RGB_DIR_PATH, masks_dir=config.SOURCE_MASKS_DIR_PATH,
                           depth_dir=config.SOURCE_DEPTH_DIR_PATH,
                           extend_dataset=True, num_images=config.ITERATIONS,
                           scale=img_scale, apply_imgaug=config.APPLY_IMAGE_AUG,
                           take_center_crop=config.TAKE_CENTER_CROP, crop_h=config.CROP_H, crop_w=config.CROP_W,
                           mask_suffix=config.SOURCE_GT_MASK_SUFFIX, depth_suffix=config.SOURCE_DEPTH_SUFFIX)

    source_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    source_train_iterator = enumerate(source_train_loader)

    ##########################
    # creating real dataset
    ##########################
    dataset = BasicDataset(imgs_dir=config.TARGET_RGB_DIR_PATH, masks_dir=config.TARGET_MASKS_DIR_PATH,
                           depth_dir=config.TARGET_DEPTH_DIR_PATH,
                           extend_dataset=True, num_images=int(config.ITERATIONS + config.NUM_VAL),
                           scale=img_scale, apply_imgaug=config.APPLY_IMAGE_AUG,
                           take_center_crop=config.TAKE_CENTER_CROP, crop_h=config.CROP_H, crop_w=config.CROP_W,
                           mask_suffix=config.TARGET_GT_MASK_SUFFIX, depth_suffix=config.TARGET_DEPTH_SUFFIX)

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

    writer = SummaryWriter(comment=f'CLAN_ADV_{config.EXPERIMENT}')
    global_step = 0

    logging.info(f'''Starting training:
        Model:           {config.MODEL_SELECTION}
        Multi Pred:      {config.MULTI_PRED}
        Use Depth:       {config.USE_DEPTH_IMAGES}
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
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.99))
    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    # if config.GAN == 'Vanilla':
    #     bce_loss = torch.nn.BCEWithLogitsLoss()
    # elif config.GAN == 'LS':
    #     bce_loss = torch.nn.MSELoss()
    # else:
    #     raise NotImplementedError

    ##################
    # TODO: LR scheduler
    ##################

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_warmup(base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def adjust_learning_rate(optimizer, i_iter, lr=config.LR):
        if i_iter < config.PREHEAT_STEPS:
            lr = lr_warmup(lr, i_iter, config.PREHEAT_STEPS)
        else:
            lr = lr_poly(lr, i_iter, config.ITERATIONS, config.POWER)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def adjust_learning_rate_D(optimizer, i_iter, lr=config.LR):
        if i_iter < config.PREHEAT_STEPS:
            lr = lr_warmup(lr, i_iter, config.PREHEAT_STEPS)
        else:
            lr = lr_poly(lr, i_iter, config.ITERATIONS, config.POWER)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def weightmap(pred1, pred2):
        output = 1.0 - torch.sum((pred1 * pred2), 1).view(-1, 1, pred1.size(2), pred1.size(3)) / \
                 (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(-1, 1, pred1.size(2), pred1.size(3))
        return output

    ##################
    ##################

    net.train()
    discriminator.train()
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
        # for iteration in range(config.ITERATIONS):
        while global_step < config.ITERATIONS:

            seg_loss = 0
            seg_adv_loss = 0
            weight_loss = 0
            dis_loss_source = 0
            dis_loss_target = 0

            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()

            adjust_learning_rate(optimizer, global_step)
            adjust_learning_rate_D(optimizer_discriminator, global_step)

            damping = (1 - global_step / config.ITERATIONS)

            ##########################
            # Seg w./ Source
            ##########################
            _, batch = source_train_iterator.__next__()
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
                    masks_pred1_source, masks_pred2_source = net(imgs, depths)
                else:
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
            for param in discriminator.parameters():
                param.requires_grad = False

            ###############
            ###############
            _, batch = target_train_iterator.__next__()
            imgs = batch['image']
            depths = batch['depth']
            imgs = imgs.to(device=device, dtype=torch.float32)
            depths = depths.to(device=device, dtype=torch.float32)
            # true_masks = batch['mask']
            # true_masks = true_masks.to(device=device, dtype=mask_type)

            if config.MULTI_PRED:
                ###################################
                # multi
                ###################################
                if config.USE_DEPTH_IMAGES:
                    masks_pred1_target, masks_pred2_target = net(imgs, depths)
                else:
                    masks_pred1_target, masks_pred2_target = net(imgs)

            discriminator_out1_target = upsample(discriminator(F.softmax(masks_pred1_target + masks_pred2_target, dim = 1)))
            discriminator_adv_fill = Variable(
                torch.FloatTensor(discriminator_out1_target.data.size()).fill_(source_label)).to(device=device)

            weight_map = weightmap(F.softmax(masks_pred1_target, dim=1), F.softmax(masks_pred2_target, dim=1))

            # Adaptive Adversarial Loss
            if (global_step > config.PREHEAT_STEPS):
                loss_adv = weighted_bce_loss(discriminator_out1_target, discriminator_adv_fill,
                                             weight_map, config.EPSILON, config.LAMDA_LOCAL)
            else:
                loss_adv = bce_loss(discriminator_out1_target, discriminator_adv_fill)

            loss_adv = loss_adv * config.LAMDA_ADV * damping
            seg_adv_loss += loss_adv.item()
            loss_adv.backward()

            writer.add_scalar('Adv_Loss/Adv', seg_adv_loss, global_step)

            #############################
            # Weight Discrepancy Loss
            #############################

            W5 = None
            W6 = None
            for (w5, w6) in zip(net.last_conv1.parameters(), net.last_conv2.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)

            loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
            weight_loss = loss_weight.item() * config.LAMDA_WEIGHT * damping * 2
            loss_weight.backward()

            writer.add_scalar('Adv_Loss/Weights', weight_loss, global_step)

            #############################
            #############################

            # now accumulate grads in D
            for param in discriminator.parameters():
                param.requires_grad = True

            #############################
            # DISCRIMINATOR w/ Source
            #############################
            masks_pred1_source = masks_pred1_source.detach()
            masks_pred2_source = masks_pred2_source.detach()

            discriminator_out_source = upsample(discriminator(F.softmax(masks_pred1_source + masks_pred2_source, dim=1)))
            discriminator_fill_source = Variable(
                torch.FloatTensor(discriminator_out_source.data.size()).fill_(source_label)).to(device=device)

            loss_D_source = bce_loss(discriminator_out_source, discriminator_fill_source)
            dis_loss_source += loss_D_source.item()
            loss_D_source.backward()

            #############################
            # DISCRIMINATOR w/ Target
            #############################

            masks_pred1_target = masks_pred1_target.detach()
            masks_pred2_target = masks_pred2_target.detach()

            discriminator_out_target = upsample(discriminator(F.softmax(masks_pred1_target + masks_pred1_target, dim=1)))
            discriminator_fill_target = Variable(
                torch.FloatTensor(discriminator_out_target.data.size()).fill_(target_label)).to(device=device)

            # Adaptive Adversarial Loss
            if (global_step > config.PREHEAT_STEPS):
                loss_D_t = weighted_bce_loss(discriminator_out_target, discriminator_fill_target,
                                             weight_map, config.EPSILON, config.LAMDA_LOCAL)
            else:
                loss_D_t = bce_loss(discriminator_out_target, discriminator_fill_target)

            loss_D_target = bce_loss(discriminator_out_target, discriminator_fill_target)
            dis_loss_target += loss_D_target.item()
            loss_D_target.backward()

            writer.add_scalar('Adv_Loss/Discriminator_Source', dis_loss_source, global_step)
            writer.add_scalar('Adv_Loss/Discriminator_Target', dis_loss_target, global_step)

            ##########################
            ##########################

            optimizer.step()
            optimizer_discriminator.step()

            pbar.set_postfix(**{'loss ': seg_loss,
                                'adv_loss ': seg_adv_loss,
                                'dis_loss_s ': dis_loss_source,
                                'dis_loss_t ': dis_loss_target})

            global_step += 1
            pbar.update(imgs.shape[0])

            global_step += 1
            if global_step % (config.NUM_IMAGES_PER_EPOCH // (1)) == 0:
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
                for tag, value in discriminator.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        print('Dis1_Layer: ', tag.split('/'))
                        writer.add_histogram('dis_weights1/' + tag, value.data.cpu().numpy(), global_step)
                        pass
                    else:
                        writer.add_histogram('dis_weights1/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('dis_grads1/' + tag, value.grad.data.cpu().numpy(), global_step)

                # weighted fwb score
                val_loss, Fwb = eval_net(net, upsample, target_val_loader, writer, best_Fwb, global_step, device)
                writer.add_scalar('Weighted-Fb/Current-Fwb', Fwb, global_step)

                # scheduler
                writer.add_scalar('learning_rate/seg', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('learning_rate/dis1', optimizer_discriminator.param_groups[0]['lr'], global_step)

                if config.NUM_CLASSES > 1:
                    writer.add_scalar('Loss/test', val_loss, global_step)
                else:
                    logging.info('Validation Dice Coeff: {}'.format(Fwb))
                    writer.add_scalar('Dice/test', Fwb, global_step)

                if Fwb > best_Fwb and save_cp:
                    best_Fwb = Fwb
                    writer.add_scalar('Weighted-Fb/Best-Fwb', best_Fwb, global_step)
                    torch.save(net.state_dict(), config.BEST_MODEL_SAVE_PATH)
                    torch.save(discriminator.state_dict(), config.BEST_DIS1_SAVE_PATH)
                    logging.info('Best Model Saved with Fwb: {:.5}!'.format(best_Fwb))

    if save_cp:
        torch.save(net.state_dict(), config.MODEL_SAVE_PATH + "Epoch_{}_Best_Seg_{:.5}.pth".format((config.EPOCHS, best_Fwb)))
        torch.save(discriminator.state_dict(), config.MODEL_SAVE_PATH + "Epoch_{}_Best_Dis1_{:.5}.pth".format((config.EPOCHS, best_Fwb)))
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
    # faster convolutions, but more memoryc
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
    # Discriminator
    #######################

    discriminator = FCDiscriminator(num_classes=config.NUM_CLASSES)
    discriminator.to(device=device)

    from torchsummary import summary
    summary(discriminator, (config.NUM_CLASSES, config.CROP_H, config.CROP_W))

    logging.info(f'Discriminator:\n'
                 f'\t{config.NUM_CLASSES} input channels (classes)\n')

    if config.DIS_SAVED_WEIGHTS:
        discriminator.load_state_dict(torch.load(config.DIS_SAVED_WEIGHTS, map_location=device))
        logging.info(f'Discriminator loaded from {config.DIS_SAVED_WEIGHTS}')
    else:
        logging.info(f'Training Discriminator from scratch!\n')

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
