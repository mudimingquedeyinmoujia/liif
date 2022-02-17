import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('facescape-liif-wrapper')
class FacescapeWrapper(Dataset):

    def __init__(self, dataset, inp_size=64, scale_min=1, scale_max=None, random_sample=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.random_sample = random_sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_path, img_id, exp, index = self.dataset[idx]
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        s = random.uniform(self.scale_min, self.scale_max)
        gt_size = round(self.inp_size * s)
        gt = resize_fn(img, gt_size)
        inp= resize_fn(img,self.inp_size)

        hr_coord, hr_rgb = to_pixel_samples(gt.contiguous())

        if self.random_sample is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.random_sample, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / gt_size
        cell[:, 1] *= 2 / gt_size

        return {
            'inp': inp, # 4096 scale to inp_size=64
            'coord': hr_coord, # 4096 scale to inp_size*s
            'cell': cell,
            'gt': hr_rgb # 4096 scale to inp_size*s
        }


@register('facescape-valid-wrapper')
class FacescapeValidWrapper(Dataset):

    def __init__(self, dataset, inp_size=64, scale_min=1, scale_max=None, random_sample=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.random_sample = random_sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_path, img_id, exp, index = self.dataset[idx]
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        s = random.uniform(self.scale_min, self.scale_max)
        gt_size = round(self.inp_size * s)
        gt = resize_fn(img, gt_size)
        inp= resize_fn(img,self.inp_size)

        hr_coord, hr_rgb = to_pixel_samples(gt.contiguous())

        if self.random_sample is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.random_sample, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / gt_size
        cell[:, 1] *= 2 / gt_size

        return {
            'inp': inp, # 4096 scale to inp_size=64 [3,64,64]
            'coord': hr_coord, # 4096 scale to inp_size*s [sample,2]
            'cell': cell,
            'gt': hr_rgb, # 4096 scale to inp_size*s [sample,3]
            'gt512':resize_fn(img,512) # [3,512,512]
        }