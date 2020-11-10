from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)

#######################################
#######################################

'''
Model Selection:
'og_unet'
'smp_unet'
'smp_fpn'
'og_deeplab'
'pretrained_deeplab'
'pretrained_deeplab_multi'
'pretrained_deeplab_multi_depth'
'''
MULTI_PRED = True
USE_DEPTH_IMAGES = True
MODEL_SELECTION = 'pretrained_deeplab_multi_depth'

BACKBONE = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'

SEG_SAVED_WEIGHTS = ''
DIS_SAVED_WEIGHTS = ''
# SEG_SAVED_WEIGHTS = '/home/akeaveny/catkin_ws/src/Pytorch-UNet/checkpoints/umd/segmentation/pretrained_deeplab_multi_hammer_syn/BEST_SEG_MODEL.pth'
# DIS_SAVED_WEIGHTS = '/home/akeaveny/catkin_ws/src/Pytorch-UNet/checkpoints/umd/discriminator/pretrained_deeplab_multi_CLAN_hammer_syn/BEST_DIS1_MODEL.pth'

LAMBDA_SEG = 1
ADV_SEG1 = 1
ADV_SEG2 = 1

POWER = 0.9
LAMDA_WEIGHT = 0.01
LAMDA_ADV = 0.001
LAMDA_LOCAL = 40
EPSILON = 0.4

GAN = 'Vanilla'

EPOCHS = 2
BATCH_SIZE = 2
NUM_IMAGES_PER_EPOCH = 100
ITERATIONS = int(EPOCHS * NUM_IMAGES_PER_EPOCH)
PREHEAT_STEPS = int(ITERATIONS/20)

LR = 0.0001

#######################################
#######################################

# EXPERIMENT = MODEL_SELECTION + '_hammer' + '_syn/'
EXPERIMENT = MODEL_SELECTION + '_CLAN' + '_hammer' + '_syn/'
# CHECKPOINT_DIR_PATH = str(ROOT_DIR_PATH) + '/checkpoints/' + 'umd/' + 'segmentation/' + EXPERIMENT
CHECKPOINT_DIR_PATH = str(ROOT_DIR_PATH) + '/checkpoints/' + 'umd/' + 'discriminator/' + EXPERIMENT
MODEL_SAVE_PATH = str(CHECKPOINT_DIR_PATH)
BEST_MODEL_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_SEG_MODEL.pth'
BEST_DIS1_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_DIS1_MODEL.pth'
BEST_DIS2_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_DIS2_MODEL.pth'

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
NUM_TRAIN = NUM_IMAGES_PER_EPOCH
NUM_VAL = int(NUM_IMAGES_PER_EPOCH * 10/100)
NUM_TEST = 25

#######################################
# SEGMENTATION
#######################################

## syn
DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
RGB_IMG_EXT = '.png'
DEPTH_SUFFIX = '_depth'
DEPTH_IMG_EXT = '_depth.png'
GT_MASK_EXT = '_gt_affordance.png'
GT_MASK_SUFFIX = '_gt_affordance'

### real
# DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/real/objects/hammer/'
# TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
# RGB_IMG_EXT = '.jpg'
# DEPTH_SUFFIX = '_depth'
# DEPTH_IMG_EXT = '_depth.png'
# GT_MASK_EXT = '_label.png'
# GT_MASK_SUFFIX = '_label'

GT_SAVE_MASK_EXT = '_gt.png'
PRED_MASK_EXT = '_pred.png'

RGB_DIR_PATH = DATASET_DIR_PATH + 'train/rgb/'
DEPTH_DIR_PATH = DATASET_DIR_PATH + 'train/depth/'
MASKS_DIR_PATH = DATASET_DIR_PATH + 'train/masks/'
VAL_PRED_DIR_PATH = DATASET_DIR_PATH + 'train/pred/'

TEST_RGB_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/rgb/'
TEST_DEPTH_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/depth/'
TEST_MASKS_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/masks/'
TEST_PRED_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/pred/'

#######################################
# ADAPT SEG NET
#######################################

### source
SOURCE_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
SOURCE_RGB_IMG_EXT = '.png'
SOURCE_DEPTH_SUFFIX = '_depth'
SOURCE_DEPTH_IMG_EXT = '_depth.png'
SOURCE_GT_MASK_EXT = '_gt_affordance.png'
SOURCE_GT_MASK_SUFFIX = '_gt_affordance'

SOURCE_RGB_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/rgb/'
SOURCE_DEPTH_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/depth/'
SOURCE_MASKS_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/masks/'

### real
TARGET_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/real/objects/hammer/'
TARGET_RGB_IMG_EXT = '.jpg'
TARGET_DEPTH_SUFFIX = '_depth'
TARGET_DEPTH_IMG_EXT = '_depth.png'
TARGET_GT_MASK_EXT = '_label.png'
TARGET_GT_MASK_SUFFIX = '_label'

GT_SAVE_MASK_EXT = '_gt.png'
PRED_MASK_EXT = '_pred.png'

TARGET_RGB_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/rgb/'
TARGET_DEPTH_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/depth/'
TARGET_MASKS_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/masks/'
TARGET_PRED_DIR_PATH = TARGET_DATASET_DIR_PATH + 'test/pred/'