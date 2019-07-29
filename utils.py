import os
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm, tqdm_notebook
import cv2
from collections import OrderedDict, Counter
import json
import yaml

from torch.utils.data import Dataset
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import albumentations
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from dataset import RetinopathyDataset


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
#TRAIN_CSV_PATH = os.path.join(DIR_PATH, 'data/train.csv')
TRAIN_CSV_PATH = os.path.join(DIR_PATH, 'data/train_all.csv')
TEST_CSV_PATH = os.path.join(DIR_PATH, 'data/test.csv')
#TRAIN_DIR_PATH = os.path.join(DIR_PATH, 'data/train_images')
#TRAIN_DIR_PATH = os.path.join(DIR_PATH, 'data/train_all')
#TRAIN_DIR_PATH = os.path.join(DIR_PATH, 'data/train_all_crop')
TRAIN_DIR_PATH = os.path.join(DIR_PATH, 'data/train_all_crop_blur30')
TEST_DIR_PATH = os.path.join(DIR_PATH, 'data/test_images')
SAMPLE_SUBMISSION_PATH = os.path.join(DIR_PATH, 'data/sample_submission.csv')
RESULT_DIR = os.path.join(DIR_PATH, 'results')
N_CLASS = 5
N_CLASS_REG = 1



def run_model(df, train_index, valid_index, epochs, batch_size, image_size, model_name, n_class,
              optimizer_name, loss_name, lr, task, multi,
              device, model_path, num_workers, lr_scheduler, 
              azure_run=None, writer=None):
    
    train_tfms = build_transform(size=image_size, mode='train')
    val_tfms = build_transform(size=image_size, mode='test')
    
    train = df.iloc[train_index]
    valid = df.iloc[valid_index]
    train_dataset = RetinopathyDataset(df=train, mode='train', 
                                       transform=train_tfms)
    val_dataset = RetinopathyDataset(df=valid, mode='train', 
                                     transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             shuffle=False, pin_memory=True, num_workers=num_workers)

    model = build_model(model_name, n_class)
    if multi:
        model = nn.DataParallel(model)

    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=lr/100)
    if lr_scheduler:
        scheduler = build_lr_scheduler(optimizer, lr_scheduler['name'], **lr_scheduler['option'])
    loss_func = build_loss(loss_name)


    model.to(device)
    best_val_score = 1000000
    for epoch in tqdm(range(epochs)):
        model.train()
        avg_loss = 0.
        # train
        for data, target in tqdm(train_loader):
            data = data.to(device)
            if task == 'class':
                target = target.to(device)
            elif task == 'reg':
                target = target.view(-1, 1)
                target = target.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            y_pred = model(data)
            loss = loss_func(y_pred, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(train_loader)

        # validation
        model.eval()
        avg_val_loss = 0.
        valid_preds_fold = np.zeros((len(val_dataset), n_class))
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            if task == 'class':
                target = target.to(device)
            elif task == 'reg':
                target = target.view(-1, 1)
                target = target.to(device, dtype=torch.float)
            with torch.no_grad():
                y_pred = model(data).detach()

            avg_val_loss += loss_func(y_pred, target).item()
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
        avg_val_loss /= len(val_loader)
        
        if task == 'class':
            round_valid_preds = np.argmax(valid_preds_fold, axis=1)
        elif task == 'reg':
            initial_coef = [0.5, 1.5, 2.5, 3.5]
            optR = OptimizedRounder()
            round_valid_preds = optR.predict(valid_preds_fold, initial_coef)
        val_kappa = cohen_kappa_score(round_valid_preds,
                                      val_dataset.df['diagnosis'],
                                      weights='quadratic')
        print(f'epoch {epoch+1} / {epochs}, loss: {avg_loss:.5}, val_loss: {avg_val_loss:.5}, val_kappa: {val_kappa:.5}')

        # step scheduler
        if scheduler:
            scheduler.step()

        if writer:
            writer.add_scalar('loss', avg_loss, epoch)
            writer.add_scalar('val_loss', avg_val_loss, epoch)
            writer.add_scalar('val_kappa', val_kappa, epoch)
        if azure_run:
            azure_run.log('loss', avg_loss)
            azure_run.log('val loss', avg_val_loss)
            azure_run.log('val kappa', val_kappa)
            if scheduler:
                azure_run.log('lr', scheduler.get_lr()[0])

        is_best = bool(avg_val_loss < best_val_score)
        if is_best:
            best_val_score = avg_val_loss
            print(f'update best score !! current best score: {best_val_score:.5} !!')
        save_checkpoint(model, is_best, model_path)

    fold_best_model = load_pytorch_model(model_name, model_path, n_class)
    if multi:
        fold_best_model = nn.DataParallel(fold_best_model)
    best_valid_preds = predict(fold_best_model, val_loader, n_class, device)
    
    return best_valid_preds, valid['diagnosis']
        
    
    
    
def predict(model, dataloader, n_class, device):
    model.eval()
    model.to(device)
    preds = np.zeros([0, n_class])
    for data, _ in dataloader:
        data = data.to(device)
        with torch.no_grad():
            y_pred = model(data).detach()
        #y_pred = F.softmax(y_pred, dim=1).cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        preds = np.concatenate([preds, y_pred])
    return preds
    
    
def build_model(model_name, n_class=N_CLASS, pretrained=True):
    if model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, n_class)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, n_class)
    elif model_name == 'efficientnet-b2':
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_name(model_name)
        model._fc = nn.Linear(1408, n_class)
    elif model_name == 'efficientnet-b3':
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_name(model_name)
        model._fc = nn.Linear(1536, n_class)
    elif model_name == 'efficientnet-b4':
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_name(model_name)
        model._fc = nn.Linear(1792, n_class)
        
        
    elif model_name == 'se_resnext50_32x4d':
        if pretrained:
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, 
                                                          pretrained='imagenet')
        else:
            model = pretrainedmodels.__dict__[model_name](num_classes=1000)
            
        model.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model.last_linear = nn.Linear(2048, n_class)
    else:
        raise ValueError('unknown model name')
    return model

def build_optimizer(optimizer_name, *args, **kwargs):
    if optimizer_name == 'adam':
        optimizer = optim.Adam(*args, **kwargs)
    else:
        raise ValueError('unknown optimizer name')
    return optimizer

def build_loss(loss_name):
    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError('unknown loss name')

def build_lr_scheduler(optimizer, scheduler_name, *args, **kwargs):
    
    if scheduler_name == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, *args, **kwargs)
    elif scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, *args, **kwargs)
    return scheduler
    
    
def build_transform(size, mode):
    border_mode = cv2.BORDER_CONSTANT
    if mode == 'train':
        transform = albumentations.Compose([
            albumentations.Resize(size, size),
            albumentations.Flip(),
            albumentations.Rotate(border_mode=border_mode),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10, 
                                            border_mode=border_mode),
            albumentations.RandomBrightness(limit=0.5),
            albumentations.RandomContrast(limit=0.3),
            albumentations.RandomGamma(),
            albumentations.Normalize(),
        ])
    elif mode == 'test':
        transform = albumentations.Compose([
            albumentations.Resize(size, size),
            albumentations.Normalize(),
        ])
    return transform

def save_checkpoint(model, is_best, path):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print (f"=> Saving a new best to {path}")
        save_pytorch_model(model, path)  # save checkpoint
    else:
        print ("=> Validation metric did not improve")
        
        
def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_pytorch_model(model_name, path, n_class, *args, **kwargs):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model = build_model(model_name, n_class, pretrained=False)
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(state_dict)
    return model

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
    
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef,
                                          method='nelder-mead')
        
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

    
    
    
def load_yaml(path):
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

def save_yaml(path, yaml_data):
    with open(path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    
    
    
    