import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

rootpath='./datasets/div2k/DIV2K_train_HR'
filenames=sorted(os.listdir(rootpath))
files=[]
for filename in filenames:
    file=os.path.join(rootpath,filename)
    files.append(file)

for file in files:
    img=transforms.ToTensor()(Image.open(file).convert('RGB'))
    print(img.shape)