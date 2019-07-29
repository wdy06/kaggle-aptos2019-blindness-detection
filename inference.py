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

from azureml.core import Workspace, Experiment
from torch.utils.tensorboard import SummaryWriter


import utils
from dataset import RetinopathyDataset

parser = argparse.ArgumentParser(description='aptos2019 blindness detection on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--cv", help="do cross validation",
                    action="store_true")
parser.add_argument("--multi", help="use multi gpu",
                    action="store_true")
parser.add_argument('--model', '-m', type=str,
                    help='cnn model name')
parser.add_argument('--batch', '-B', type=int, default=64,
                    help='batch size')
parser.add_argument('--size', '-S', type=int, default=256,
                    help='image size')
parser.add_argument('--weight', '-w', type=str,
                    help='path to model weight')
parser.add_argument('--task', '-t', type=str, default='class',
                    help='task type: class or reg')
args = parser.parse_args()

def main():
    N_FOLDS = 5
    BATCH_SIZE = args.batch
    IMAGE_SIZE = args.size
    model_name = args.model
    image_size = args.size
    num_workers = 64
    
    assert args.task in ['class', 'reg']
    if args.task == 'class':
        n_class = utils.N_CLASS
    elif args.task == 'reg':
        n_class = utils.N_CLASS_REG
    
    print(f'found {torch.cuda.device_count()} gpus !!')
    if args.multi:
        print('use multi gpu !!')



    device = torch.device("cuda:0")
    train_df = pd.read_csv(utils.TRAIN_CSV_PATH)
    if args.debug:
        train_df = train_df[:1000]

    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=41, shuffle=True)
    indices = list(skf.split(train_df, train_df['diagnosis']))
    if not args.cv:
        print('do not use cross validation')
        #indices = [indices[0]]
        with open('folds_index.json', 'rb') as f:
            indices = pickle.load(f)
        

    # cross validation
    oof_preds = np.zeros((len(train_df), utils.N_CLASS))
    val_tfms = utils.build_transform(size=image_size, mode='test')
    for i_fold, (train_index, valid_index) in tqdm(enumerate(indices)):
        valid = train_df.iloc[valid_index]
        val_dataset = RetinopathyDataset(df=valid, mode='train', 
                                         transform=val_tfms)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                                                 shuffle=False, pin_memory=True, num_workers=num_workers)
        model_path = os.path.join(args.weight, f'model_fold{i_fold}')
        model = utils.load_pytorch_model(model_name, model_path, n_class)
        if args.multi:
            model = nn.DataParallel(model)
        y_pred = utils.predict(model, val_loader, n_class, device)
        if args.cv:
            oof_preds[valid_index] = y_pred
    if args.cv:
        valid_preds = oof_preds
        valid_true = train_df['diagnosis']
#         val_kappa = cohen_kappa_score(np.argmax(oof_preds, axis=1), train_df['diagnosis'],
#                                       weights='quadratic')
    else:
        valid_preds = y_pred
        valid_true = valid['diagnosis']
#         val_kappa = cohen_kappa_score(np.argmax(y_pred, axis=1), valid['diagnosis'],
#                                       weights='quadratic')
    if args.task == 'class':
        round_valid_preds = np.argmax(valid_preds, axis=1)
    elif args.task == 'reg':
        print('optimizing threshold ...')
        optR = utils.OptimizedRounder()
        optR.fit(valid_preds, valid_true)
        coef = optR.coefficients()
        print(f'best coef: {coef}')
        round_valid_preds = optR.predict(valid_preds, coef)
    val_kappa = cohen_kappa_score(round_valid_preds, valid_true, weights='quadratic')

        
    print(f'best val kappa: {val_kappa}')

#     test_csv = pd.read_csv(utils.TEST_CSV_PATH)
#     test_tfms = utils.build_transform(size=IMAGE_SIZE, mode='test')
#     test_dataset = RetinopathyDataset(df=test_csv, mode='test', transform=test_tfms,
#                                       auto_crop=True, add_blur=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
#                                               shuffle=False, pin_memory=True,
#                                               num_workers=num_workers)

#     test_preds = np.zeros((len(test_csv), utils.N_CLASS))
#     for i in range(len(indices)):
#         model = utils.load_pytorch_model(model_name, os.path.join(result_dir, f'model_fold{i}'))
#         test_preds += utils.predict(model, test_loader, n_class=5, device=device)

#     submission_csv = pd.read_csv(utils.SAMPLE_SUBMISSION_PATH)
#     submission_csv['diagnosis'] = np.argmax(test_preds, axis=1)
#     submission_csv.to_csv(os.path.join(result_dir, 'submission.csv'),
#                           index=False)

    print('finish!!!')
    
if __name__ == "__main__":
    main()
    
    
    
    
    