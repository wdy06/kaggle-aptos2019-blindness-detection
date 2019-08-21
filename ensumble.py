import numpy as np 
import os
import random as rn

#fix random seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
import torch
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.deterministic = True

import argparse
import pandas as pd 
import gc
import logging
from datetime import datetime
import warnings
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import pickle
from scipy.special import softmax

from azureml.core import Workspace, Experiment
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F


import utils
from dataset import RetinopathyDataset

parser = argparse.ArgumentParser(description='aptos2019 blindness detection on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--multi", help="use multi gpu",
                    action="store_true")
parser.add_argument('--batch', '-B', type=int, default=64,
                    help='batch size')
parser.add_argument('--tta', type=int, default=None,
                    help='number of tta')
args = parser.parse_args()

def main():
    N_FOLDS = 5
    BATCH_SIZE = args.batch
    num_workers = 64
    
    
    print(f'found {torch.cuda.device_count()} gpus !!')
    if args.multi:
        print('use multi gpu !!')
        
    device = torch.device("cuda:0")
    train_df = pd.read_csv(utils.TRAIN_CSV_PATH)
    if args.debug:
        train_df = train_df[:1000]

    print('do not use cross validation')
    with open('folds_index.json', 'rb') as f:
        indices = pickle.load(f)
    indices = indices[0]
        

    print(f'tta: {args.tta}')
    if args.tta:
        tfms_mode = 'test'
        num_tta = args.tta
    else:
        tfms_mode = 'val'
        num_tta = 1
        
    models = [
        {
            'path': 'results/20190725135839/model_fold0',
            'model_name': 'efficientnet-b3',
            'image_size': 256,
            'task': 'class'
        },
        {
            'path': 'results/20190804235524/model_fold0',
            'model_name': 'efficientnet-b3',
            'image_size': 320,
            'task': 'class'
        },
        {
            'path': 'results/20190805111653/model_fold0',
            'model_name': 'efficientnet-b3',
            'image_size': 512,
            'task': 'class'
        },
        {
            'path': 'results/20190723033442/model_fold0',
            'model_name': 'se_resnext50_32x4d',
            'image_size': 256,
            'task': 'class'
        },
        {
            'path': 'results/20190723134833/model_fold0',
            'model_name': 'se_resnext50_32x4d',
            'image_size': 320,
            'task': 'class'
        },
    ]
        
    valid_preds = 0
    for model in tqdm(models):
        if model['task'] == 'class':
            n_class = utils.N_CLASS
        elif model['task'] == 'reg':
            n_class = utils.N_CLASS_REG
        val_tfms = utils.build_transform(size=model['image_size'], mode=tfms_mode)
        train_index, valid_index = indices
        valid = train_df.iloc[valid_index]
        val_dataset = RetinopathyDataset(df=valid, mode='train', 
                                         transform=val_tfms)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=BATCH_SIZE, 
                                                 shuffle=False, 
                                                 pin_memory=True,
                                                 num_workers=num_workers)
        model_path = model['path']
        model = utils.load_pytorch_model(model['model_name'], model_path, n_class)
        if args.multi:
            model = nn.DataParallel(model)
        y_pred = utils.predict(model, val_loader, n_class, device, tta=num_tta)
        #y_pred = softmax(y_pred, axis=1)
        valid_preds += y_pred
        valid_true = valid['diagnosis']
    round_valid_preds = np.argmax(valid_preds, axis=1)
    val_kappa = cohen_kappa_score(round_valid_preds, valid_true,
                                  weights='quadratic')

        
    print(f'best val kappa: {val_kappa}')


    print('finish!!!')
    
if __name__ == "__main__":
    main()
    
    
    
    
    