import os
import json
from PIL import Image
import random
import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('facescape-folder')
class FacescapeFolder(Dataset):
    """
    img: 每张4096*4096的纹理图路径
    id: 人脸纹理对应的身份id，共847个，从1-847，20张图共享同一个id，因为有20个表情
    exp: 当前人脸纹理对应的表情，共20个，从1-20
    index: 纹理图的全局索引，每张图都是唯一的，0-nums
    """

    def __init__(self, data_dir, sample_num):
        super(FacescapeFolder, self).__init__()
        self.data_dir = data_dir
        self.path_base = []
        self.total_nums = 0
        self.path_sample = []
        self.nums = 0

        for root, dirs, files in os.walk(self.data_dir, topdown=True):
            if '1_neutral.jpg' not in files:
                continue
            else:
                # check which texture dir is not 20
                # if len(files) != 60:
                #     cnt=0
                #     for i in files:
                #         if i.split('.')[-1] == 'jpg':
                #             cnt=cnt+1
                #     if cnt!=20:
                #         print(len(files),root)

                for name in files:
                    if name.split('.')[-1] == 'jpg':
                        exp_tmp = name.split('_')[0]
                        temp_path = os.path.join(root, name)
                        tid = int(root.split('\\')[-2])
                        ttid = int(root.split('\\')[-3].split('_')[-2]) - 1
                        id_tmp = ttid + tid
                        self.path_base.append([temp_path, id_tmp, exp_tmp])

        self.total_nums = len(self.path_base)
        if sample_num==-1:
            self.nums=self.total_nums
            self.path_sample=self.path_base
        else:
            self.nums = sample_num
            sam_ind = random.sample(range(0, self.total_nums), self.nums)
            self.path_sample = [self.path_base[i] for i in sam_ind]

        print('img use/find: {}/{}'.format(self.nums,self.total_nums))

    def __getitem__(self, index):
        img = self.path_sample[index][0]
        id = self.path_sample[index][1]
        exp = self.path_sample[index][2]
        return img, id, exp, index  # str int(1-847) int(1-20) int(0-nums)

    def __len__(self):
        return self.nums
