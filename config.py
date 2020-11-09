from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)

EXPERIMENT = 'pretrained_deeplab_hammer_syn/'
CHECKPOINT_DIR_PATH = str(ROOT_DIR_PATH) + '/checkpoints/' + 'umd/' + 'segmentation/' + EXPERIMENT
# CHECKPOINT_DIR_PATH = str(ROOT_DIR_PATH) + '/checkpoints/' + 'umd/' + 'discriminator/' + EXPERIMENT
MODEL_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'SEG_MODEL.pth'
BEST_MODEL_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_SEG_MODEL.pth'

#######################################
#######################################

'''
Model Selection:
'og_unet'
'smp_unet'
'smp_fpn'
'adapt_deeplab'
'pretrained_deeplab'
'''
MODEL_SELECTION = 'adapt_deeplab'
BACKBONE = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'

GAN = 'Vanilla'

EPOCHS = 50
NUM_IMAGES_PER_EPOCH = 2000
ITERATIONS = int(EPOCHS*NUM_IMAGES_PER_EPOCH)
BATCH_SIZE = 4

LR = 0.0001

#######################################
#######################################

NUM_CHANNELS = 3    # rgb images
NUM_CLASSES = 1 + 7 # background + objects

AFFORDANCE_START = 0
AFFORDANCE_END = 7

APPLY_IMAGE_AUG = True
TAKE_CENTER_CROP = True
CROP_H = 384
CROP_W = 384

TRAIN_ON_SUBSET = True
NUM_TRAIN = 1000
NUM_VAL = 100
NUM_TEST = 100

#######################################
# SEGMENTATION
#######################################

## syn
DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
RGB_IMG_EXT = '.png'
GT_MASK_EXT = '_gt_affordance.png'
GT_MASK_SUFFIX = '_gt_affordance'

## real
# DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/real/objects/hammer/'
# TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
# RGB_IMG_EXT = '.jpg'
# GT_MASK_EXT = '_label.png'
# GT_MASK_SUFFIX = '_label'

GT_SAVE_MASK_EXT = '_gt.png'
PRED_MASK_EXT = '_pred.png'

RGB_DIR_PATH = DATASET_DIR_PATH + 'train/rgb/'
MASKS_DIR_PATH = DATASET_DIR_PATH + 'train/masks/'

TEST_RGB_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/rgb/'
TEST_MASKS_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/masks/'
TEST_PRED_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/pred/'

#######################################
# ADAPT SEG NET
#######################################

### source
SOURCE_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
SOURCE_RGB_IMG_EXT = '.png'
SOURCE_GT_MASK_EXT = '_gt_affordance.png'
SOURCE_GT_MASK_SUFFIX = '_gt_affordance'

SOURCE_RGB_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/rgb/'
SOURCE_MASKS_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/masks/'

### real
TARGET_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/real/objects/hammer/'
TARGET_RGB_IMG_EXT = '.jpg'
TARGET_GT_MASK_EXT = '_label.png'
TARGET_GT_MASK_SUFFIX = '_label'

GT_SAVE_MASK_EXT = '_gt.png'
PRED_MASK_EXT = '_pred.png'

TARGET_RGB_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/rgb/'
TARGET_MASKS_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/masks/'
TARGET_PRED_DIR_PATH = TARGET_DATASET_DIR_PATH + 'test/pred/'