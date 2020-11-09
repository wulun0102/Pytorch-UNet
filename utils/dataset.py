from os.path import splitext
from os import listdir
from glob import glob

import logging

import numpy as np

from PIL import Image

import skimage.transform
from skimage.util import crop
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, extend_dataset=False, num_images=100,
                 take_center_crop=False, crop_h=384, crop_w=384, apply_imgaug=True,
                 mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.extend_dataset = extend_dataset
        self.num_images = num_images
        self.take_center_crop = take_center_crop
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Original dataset has {len(self.ids)} examples')

        # creating larger dataset
        if self.extend_dataset:
            ids = []
            total_idx = np.arange(0, len(self.ids), 1)
            for image_idx in range(self.num_images):
                idx = np.random.choice(total_idx, size=1, replace=False)
                ids.append(self.ids[int(idx)])
            self.ids = ids
        logging.info(f'Extended dataset has {len(self.ids)} examples')

        ################################
        # IMGAUG
        ################################
        self.apply_imgaug = apply_imgaug
        if self.apply_imgaug:
            # create the augmentation sequences
            self.affine = iaa.Sequential([
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.5),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
            ], random_order=True)

            self.colour = iaa.Sometimes(0.833, iaa.Sequential([
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5,
                              iaa.GaussianBlur(sigma=(0, 0.5))
                              ),
                # Strengthen or weaken the contrast in each image.
                iaa.contrast.LinearContrast((0.75, 1.25)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
            ], random_order=True))  # apply augmenters in random order

    def __len__(self):
        return len(self.ids)

    @classmethod
    def crop(cls, pil_img, crop_w, crop_h, is_img=True):
        img_width, img_height = pil_img.size
        left, right = (img_width - crop_w) / 2, (img_width + crop_w) / 2
        top, bottom = (img_height - crop_h) / 2, (img_height + crop_h) / 2
        left, top = round(max(0, left)), round(max(0, top))
        right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
        # pil_img = pil_img.crop((left, top, right, bottom)).resize((crop_w, crop_h))
        pil_img = pil_img.crop((left, top, right, bottom))
        if is_img:
            img_channels = np.array(pil_img).shape[-1]
            img_channels = 3 if img_channels == 4 else img_channels
            resize_img = np.zeros((crop_w, crop_h, img_channels))
            resize_img[0:(bottom-top), 0:(right - left), :img_channels] = np.array(pil_img)[..., :img_channels]
        else:
            resize_img = np.zeros((crop_w, crop_h))
            resize_img[0:(bottom-top), 0:(right - left)] = np.array(pil_img)
        resize_img = np.array(resize_img, dtype=np.uint8)

        return resize_img

    @classmethod
    def preprocess(cls, pil_img, scale, is_img=True):
        h, w = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img_nd = np.array(pil_img)
        img_nd = skimage.transform.resize(img_nd,
                                           (newW, newH),
                                           mode='edge',
                                           anti_aliasing=False,
                                           anti_aliasing_sigma=None,
                                           preserve_range=True,
                                           order=0)


        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if img_nd.shape[-1] == 4:
            img_nd = img_nd[..., :3]

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if is_img:
            if img_trans.max() > 1:
                img_trans = img_trans / 255
        #else:
        #    img_trans = rgb2gray(img_trans.transpose((1, 2, 0)))

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        #print('!!!!!!!!!!!!!!!', mask_file, '!!!!!!!!!!!!!!!!!')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        ################################
        ### TODO: IMGAUG
        ################################
        if self.take_center_crop:
            img = self.crop(img,self.crop_w, self.crop_w, is_img=True)
            mask = self.crop(mask, self.crop_w, self.crop_w, is_img=False)

        if self.apply_imgaug:
            img, mask = np.array(img), np.array(mask)
            segmap = SegmentationMapsOnImage(mask, shape=np.array(img).shape)
            image, segmap = self.affine(image=img, segmentation_maps=segmap)
            mask = segmap.get_arr()

            img = self.colour(image=image)
            img, mask = Image.fromarray(img), Image.fromarray(mask)

        img = self.preprocess(img, self.scale, is_img=True)
        mask = self.preprocess(mask, self.scale, is_img=False)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

#######################################
#######################################

# class CarvanaDataset(BasicDataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1):
#         super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

if __name__ == '__main__':

    ##############
    ##############

    DATA_DIR = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
    RGB_DATA_DIR = DATA_DIR + 'train/rgb/'
    LABELS_DATA_DIR = DATA_DIR + 'train/masks/'

    ##############
    ##############

    # dst = UMDSynDataset(imgs_dir=RGB_DATA_DIR, masks_dir=LABELS_DATA_DIR, scale=1)
    dst = BasicDataset(imgs_dir=RGB_DATA_DIR, masks_dir=LABELS_DATA_DIR,
                       scale=1, take_center_crop=True, crop_h=384, crop_w=384, apply_imgaug=True,
                       mask_suffix='_gt_affordance')

    trainloader = data.DataLoader(dst, batch_size=1)
    print("\n\n***** len dataloader: {} *****\n\n".format(len(trainloader)))
    for i, data in enumerate(trainloader):
        imgs, labels = data['image'], data['mask']
        if i >= 0:
            ### img
            img = np.array(data['image'].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0))
            print("*** IMG:\t\t dtype: {},\t size: {} ***".format(img.dtype, img.shape))
            ### label
            labels = np.squeeze(np.array(data['mask'].squeeze(0).detach().cpu().numpy()))
            print("*** LABEL:\t\t aff_id: {},\t dtype: {},\t size: {} ***".format(np.unique(labels), labels.dtype, labels.shape))
            ### plot
            plt.figure(0)
            plt.subplot(1, 2, 1)
            plt.title("img")
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.title("labels")
            plt.imshow(np.array(labels))
            plt.show()
            plt.ioff()