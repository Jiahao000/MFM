import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data.transforms import _pil_interp

from .random_degradations import RandomBlur, RandomNoise


class FreqMaskGenerator:
    def __init__(self,
                 input_size=224,
                 mask_radius1=16,
                 mask_radius2=999,
                 sample_ratio=0.5):
        self.input_size = input_size
        self.mask_radius1 = mask_radius1
        self.mask_radius2 = mask_radius2
        self.sample_ratio = sample_ratio
        self.mask = np.ones((self.input_size, self.input_size), dtype=int)
        for y in range(self.input_size):
            for x in range(self.input_size):
                if ((x - self.input_size // 2) ** 2 + (y - self.input_size // 2) ** 2) >= self.mask_radius1 ** 2 \
                        and ((x - self.input_size // 2) ** 2 + (y - self.input_size // 2) ** 2) < self.mask_radius2 ** 2:
                    self.mask[y, x] = 0

    def __call__(self):
        rnd = torch.bernoulli(torch.tensor(self.sample_ratio, dtype=torch.float)).item()
        if rnd == 0:  # high-pass
            return 1 - self.mask
        elif rnd == 1:  # low-pass
            return self.mask
        else:
            raise ValueError


class MFMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.DATA.MIN_CROP_SCALE, 1.), interpolation=_pil_interp(config.DATA.INTERPOLATION)),
            T.RandomHorizontalFlip(),
        ])

        self.filter_type = config.DATA.FILTER_TYPE
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        elif config.MODEL.TYPE == 'resnet':
            model_patch_size = 1
        else:
            raise NotImplementedError

        if config.DATA.FILTER_TYPE == 'deblur':
            self.degrade_transform = RandomBlur(
                params=dict(
                    kernel_size=config.DATA.BLUR.KERNEL_SIZE,
                    kernel_list=config.DATA.BLUR.KERNEL_LIST,
                    kernel_prob=config.DATA.BLUR.KERNEL_PROB,
                    sigma_x=config.DATA.BLUR.SIGMA_X,
                    sigma_y=config.DATA.BLUR.SIGMA_Y,
                    rotate_angle=config.DATA.BLUR.ROTATE_ANGLE,
                    beta_gaussian=config.DATA.BLUR.BETA_GAUSSIAN,
                    beta_plateau=config.DATA.BLUR.BETA_PLATEAU),
            )
        elif config.DATA.FILTER_TYPE == 'denoise':
            self.degrade_transform = RandomNoise(
                params=dict(
                    noise_type=config.DATA.NOISE.TYPE,
                    noise_prob=config.DATA.NOISE.PROB,
                    gaussian_sigma=config.DATA.NOISE.GAUSSIAN_SIGMA,
                    gaussian_gray_noise_prob=config.DATA.NOISE.GAUSSIAN_GRAY_NOISE_PROB,
                    poisson_scale=config.DATA.NOISE.POISSON_SCALE,
                    poisson_gray_noise_prob=config.DATA.NOISE.POISSON_GRAY_NOISE_PROB),
            )
        elif config.DATA.FILTER_TYPE == 'mfm':
            self.freq_mask_generator = FreqMaskGenerator(
                input_size=config.DATA.IMG_SIZE,
                mask_radius1=config.DATA.MASK_RADIUS1,
                mask_radius2=config.DATA.MASK_RADIUS2,
                sample_ratio=config.DATA.SAMPLE_RATIO
            )

    def __call__(self, img):
        img = self.transform_img(img)  # PIL Image (HxWxC, 0-255), no normalization
        if self.filter_type in ['deblur', 'denoise']:
            img_lq = np.array(img).astype(np.float32) / 255.
            img_lq = self.degrade_transform(img_lq)
            img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1))
        else:
            img_lq = None
        img = T.ToTensor()(img)  # Tensor (CxHxW, 0-1)
        if self.filter_type == 'mfm':
            mask = self.freq_mask_generator()
        else:
            mask = None
        
        return img, img_lq, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_mfm(config, logger):
    transform = MFMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader
