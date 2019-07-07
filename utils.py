import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
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
from torch.utils.tensorboard import SummaryWriter
import albumentations

from dataset import RetinopathyDataset


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(DIR_PATH, 'data/train.csv')
TEST_CSV_PATH = os.path.join(DIR_PATH, 'data/test.csv')
TRAIN_DIR_PATH = os.path.join(DIR_PATH, 'data/train_images')
TEST_DIR_PATH = os.path.join(DIR_PATH, 'data/test_images')
SAMPLE_SUBMISSION_PATH = os.path.join(DIR_PATH, 'data/sample_submission.csv')
RESULT_DIR = os.path.join(DIR_PATH, 'results')
N_CLASS = 5



def run_model(epochs, batch_size, image_size, model_name, optimizer_name, loss_name, lr,
              device, result_dir, debug, num_workers, azure_run=None, writer=None):
    
    train_csv = pd.read_csv(TRAIN_CSV_PATH)
    train, valid = train_test_split(train_csv, test_size=0.2, 
                                    stratify=train_csv['diagnosis'] , 
                                    random_state=42)
    train_tfms = build_transform(size=image_size, mode='train')
    val_tfms = build_transform(size=image_size, mode='test')
    train_dataset = RetinopathyDataset(df=train, mode='train', 
                                       transform=train_tfms, debug=debug)
    val_dataset = RetinopathyDataset(df=valid, mode='train', 
                                     transform=val_tfms, debug=debug)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             shuffle=False, pin_memory=True, num_workers=num_workers)

    model = build_model(model_name)
    
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)
    loss_func = build_loss(loss_name)
    

    save_path = os.path.join(result_dir, 'best_model')
    model.to(device)
    best_val_score = 1000000
    for epoch in tqdm(range(epochs)):
        model.train()
        avg_loss = 0.
        # train
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
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
        valid_preds_fold = np.zeros((len(val_dataset), 5))
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                y_pred = model(data).detach()

            avg_val_loss += loss_func(y_pred, target).item()
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
        avg_val_loss /= len(val_loader)
        val_kappa = cohen_kappa_score(np.argmax(valid_preds_fold, axis=1),
                                      val_dataset.df['diagnosis'],
                                      weights='quadratic')
        print(f'epoch {epoch+1} / {epochs}, loss: {avg_loss:.5}, val_loss: {avg_val_loss:.5}, val_kappa: {val_kappa:.5}')
        if writer:
            writer.add_scalar('loss', avg_loss, epoch)
            writer.add_scalar('val_loss', avg_val_loss, epoch)
            writer.add_scalar('val_kappa', val_kappa, epoch)
        if azure_run:
            azure_run.log('loss', avg_loss)
            azure_run.log('val loss', avg_val_loss)
            azure_run.log('val kappa', val_kappa)
        
        is_best = bool(avg_val_loss < best_val_score)
        if is_best:
            best_val_score = avg_val_loss
            print(f'update best score !! current best score: {best_val_score:.5} !!')
        save_checkpoint(model, is_best, save_path)
    
    
    
def predict(model, dataloader, n_class, device):
    model.eval()
    model.to(device)
    preds = np.zeros([0, n_class])
    for data, _ in dataloader:
        data = data.to(device)
        with torch.no_grad():
            y_pred = model(data).detach()
        y_pred = F.softmax(y_pred, dim=1).cpu().numpy()
        preds = np.concatenate([preds, y_pred])
    return preds
    
    
def build_model(model_name, pretrained=True):
    if model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, N_CLASS)
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
    else:
        raise ValueError('unknown loss name')

def build_transform(size, mode):
    if mode == 'train':
        transform = albumentations.Compose([
            albumentations.Resize(size, size),
            albumentations.Flip(),
            albumentations.RandomBrightness(),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
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
        print ("=> Validation Accuracy did not improve")
        
        
def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_pytorch_model(model_name, path, *args, **kwargs):
    state_dict = torch.load(path)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
    model = build_model(model_name, pretrained=False)
    #model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict)
    return model