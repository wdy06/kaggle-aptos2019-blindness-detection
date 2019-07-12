import os
import pandas as pd
import numpy as np
import cv2

from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
import cv2

from torch.utils.data import Dataset
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision import transforms
from torch.optim import lr_scheduler

import utils

class RetinopathyDataset(Dataset):

    def __init__(self, df, mode, transform, debug=False, on_memory=False):

        if debug:
            self.df = df[:100]
        else:
            self.df = df
        #self.size = size
        self.mode = mode
        if self.mode == 'train':
            dir_path = utils.TRAIN_DIR_PATH
        elif self.mode == 'test':
            dir_path = utils.TEST_DIR_PATH
        self.dir_path = dir_path
        self.transform = transform
        self.on_memory = on_memory
        images_data = []
        if on_memory:
            for i, row in tqdm(df.iterrows()):
                img_path = os.path.join(self.dir_path, row['id_code']+'.png')
                img = Image.open(img_path)
                images_data.append(img)
            self.images_data = images_data
                
                

    def __len__(self):
        return len(self.df)

#     def __getitem__(self, idx):
#         if self.on_memory:
#             image = self.images_data[idx]
#         else:
#             img_name = os.path.join(self.dir_path, self.df.iloc[idx]['id_code'] + '.png')
#             image = Image.open(img_name)
#         image = image.resize((256, 256), resample=Image.BILINEAR)
#         label = torch.tensor(self.df.iloc[idx]['diagnosis'])
#         return transforms.ToTensor()(image), label
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_path, self.df.iloc[idx]['id_code'] + '.png')
        if not os.path.exists(img_name):
            img_name = os.path.join(self.dir_path, self.df.iloc[idx]['id_code'] + '.jpeg')
        if not os.path.exists(img_name):
            print(img_name)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = image['image']
        image = image.transpose((2, 0, 1))
        if self.mode == 'train':
            label = torch.tensor(self.df.iloc[idx]['diagnosis'])
            return image, label
        else:
            return image, 'dummy'