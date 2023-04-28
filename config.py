import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Minimal crop scale
_C.DATA.MIN_CROP_SCALE = 0.2
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# Filter type, support 'mfm', 'sr', 'deblur', 'denoise'
_C.DATA.FILTER_TYPE = 'mfm'
# [MFM] Sampling ratio for low-pass filters
_C.DATA.SAMPLE_RATIO = 0.5
# [MFM] First frequency mask radius
# should be smaller than half of the image size
_C.DATA.MASK_RADIUS1 = 16
# [MFM] Second frequency mask radius
# should be larger than the first radius
# only used when masking a frequency band
# setting a larger value than the image size, e.g., 999, will have no effect
_C.DATA.MASK_RADIUS2 = 999
# [SR] SR downsampling scale factor, only used when FILTER_TYPE == 'sr'
_C.DATA.SR_FACTOR = 8
# [Deblur] Deblur parameters, only used when FILTER_TYPE == 'deblur'
_C.DATA.BLUR = CN()
_C.DATA.BLUR.KERNEL_SIZE = [7, 9, 11, 13, 15, 17, 19, 21]
_C.DATA.BLUR.KERNEL_LIST = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'sinc']
_C.DATA.BLUR.KERNEL_PROB = [0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1]
_C.DATA.BLUR.SIGMA_X = [0.2, 3]
_C.DATA.BLUR.SIGMA_Y = [0.2, 3]
_C.DATA.BLUR.ROTATE_ANGLE = [-3.1416, 3.1416]
_C.DATA.BLUR.BETA_GAUSSIAN = [0.5, 4]
_C.DATA.BLUR.BETA_PLATEAU = [1, 2]
# [Denoise] Denoise parameters, only used when FILTER_TYPE == 'denoise'
_C.DATA.NOISE = CN()
_C.DATA.NOISE.TYPE = ['gaussian', 'poisson']
_C.DATA.NOISE.PROB = [0.5, 0.5]
_C.DATA.NOISE.GAUSSIAN_SIGMA = [1, 30]
_C.DATA.NOISE.GAUSSIAN_GRAY_NOISE_PROB = 0.4
_C.DATA.NOISE.POISSON_SCALE = [0.05, 3]
_C.DATA.NOISE.POISSON_GRAY_NOISE_PROB = 0.4

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vit'
# Model name
_C.MODEL.NAME = 'pretrain'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Vision Transformer parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.IN_CHANS = 3
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.MLP_RATIO = 4
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.INIT_VALUES = 0.1
# learnable absolute positional embedding
_C.MODEL.VIT.USE_APE = True
# fixed sin-cos positional embedding
_C.MODEL.VIT.USE_FPE = False
# relative position bias
_C.MODEL.VIT.USE_RPB = False
_C.MODEL.VIT.USE_SHARED_RPB = False
_C.MODEL.VIT.USE_MEAN_POOLING = False
# Vision Transformer decoder parameters
_C.MODEL.VIT.DECODER = CN()
_C.MODEL.VIT.DECODER.EMBED_DIM = 512
_C.MODEL.VIT.DECODER.DEPTH = 0
_C.MODEL.VIT.DECODER.NUM_HEADS = 16

# ResNet parameters
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.LAYERS = [3, 4, 6, 3]
_C.MODEL.RESNET.IN_CHANS = 3

# [MFM] Reconstruction target type, support 'normal', 'masked'
_C.MODEL.RECOVER_TARGET_TYPE = 'normal'
# [MFM] Frequency loss parameters
_C.MODEL.FREQ_LOSS = CN()
_C.MODEL.FREQ_LOSS.LOSS_GAMMA = 1.
_C.MODEL.FREQ_LOSS.MATRIX_GAMMA = 1.
_C.MODEL.FREQ_LOSS.PATCH_FACTOR = 1
_C.MODEL.FREQ_LOSS.AVE_SPECTRUM = False
_C.MODEL.FREQ_LOSS.WITH_MATRIX = False
_C.MODEL.FREQ_LOSS.LOG_MATRIX = False
_C.MODEL.FREQ_LOSS.BATCH_MATRIX = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 3e-4
_C.TRAIN.WARMUP_LR = 2.5e-7
_C.TRAIN.MIN_LR = 2.5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 3.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 10
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# Path to pre-trained model
_C.PRETRAINED = ''


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('pretrained'):
        config.PRETRAINED = args.pretrained
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_custom_config(cfg):
    config = _C.clone()
    _update_config_from_file(config, cfg)
    return config
