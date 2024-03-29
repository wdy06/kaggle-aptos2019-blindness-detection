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
from pathlib import Path

from azureml.core import Workspace, Experiment
from torch.utils.tensorboard import SummaryWriter


import utils
import slack
from dataset import RetinopathyDataset

parser = argparse.ArgumentParser(description='aptos2019 blindness detection on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--cv", help="do cross validation",
                    action="store_true")
parser.add_argument("--multi", help="use multi gpu",
                    action="store_true")
parser.add_argument('--config', '-c', type=str, help='path to config')
args = parser.parse_args()

def main():
    config = utils.load_yaml(args.config)
    task = config['task']
    EPOCHS = config['epoch']
    N_FOLDS = 5
    BATCH_SIZE = config['batchsize']
    IMAGE_SIZE = config['image_size']
    model_name = config['model']
    optimizer_name = config['optimizer']
    loss = config['loss']
    lr = float(config['lr'])
    n_class = config['n_class']
    lr_scheduler = config.get('lr_scheduler')
    azure_run = None
    tb_writer = None
    num_workers = 64
    experiment_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    
    print(f'found {torch.cuda.device_count()} gpus !!')
    try:
        
        if args.debug:
            print('running in debug mode')
            EPOCHS = 1
            N_FOLDS = 2
        if args.debug:
            result_dir = Path(utils.RESULT_DIR) / ('debug-' + experiment_name)
        else:
            result_dir = Path(utils.RESULT_DIR) / experiment_name
            ws = Workspace.from_config('.aml_config/config.json')
            exp = Experiment(workspace=ws, name='kaggle-aptos2019')
            azure_run = exp.start_logging()
            azure_run.log('experiment name', experiment_name)
            azure_run.log('epoch', EPOCHS)
            azure_run.log('batch size', BATCH_SIZE)
            azure_run.log('image size', IMAGE_SIZE)
            azure_run.log('model', model_name)
            azure_run.log('optimizer', optimizer_name)
            azure_run.log('loss_name', loss['name'])
            azure_run.log('lr', lr)
            azure_run.log('lr_scheduler', lr_scheduler)
            azure_run.log('task', task)
            if args.cv:
                azure_run.log('cv', N_FOLDS)
            else:
                azure_run.log('cv', 0)
                
        if args.multi:
            print('use multi gpu !!')
            
        os.mkdir(result_dir)
        print(f'created: {result_dir}')
        utils.save_yaml(result_dir / Path(args.config).name, config)
        
        
#         if not args.debug:
#             tb_writer = SummaryWriter(log_dir=result_dir)
            

        device = torch.device("cuda:0")
        config = {
                  'epochs': EPOCHS,
                  'multi': args.multi,
                  'batch_size': BATCH_SIZE,
                  'image_size': IMAGE_SIZE,
                  'model_name': model_name,
                  'n_class': n_class,
                  'optimizer_name': optimizer_name,
                  'loss': loss,
                  'lr': lr, 
                  'lr_scheduler': lr_scheduler,
                  'task': task,
                  'device': device,
                  'num_workers': num_workers,
        }
        
        print(config)
        
        if not args.debug:
            slack.notify_start(experiment_name, config)
        train_df = pd.read_csv(utils.TRAIN_CSV_PATH)
        if args.debug:
            train_df = train_df[:1000]
        config['df'] = train_df
        
        skf = StratifiedKFold(n_splits=N_FOLDS, random_state=41, shuffle=True)
        indices = list(skf.split(train_df, train_df['diagnosis']))
        if not args.cv:
            print('do not use cross validation')
            indices = [indices[0]]
            
        # cross validation
        oof_preds = np.zeros((len(train_df), n_class))
        for i_fold, (train_index, valid_index) in tqdm(enumerate(indices)):
            model_path = result_dir / f'model_fold{i_fold}'
            config['train_index'] = train_index
            config['valid_index'] = valid_index
            config['model_path'] = str(model_path)
            if azure_run:
                if i_fold == 0:
                    config['azure_run'] = azure_run
                    y_pred, y_true = utils.run_model(**config)
                else:
                    with azure_run.child_run() as child:
                        config['azure_run'] = child
                        y_pred, y_true = utils.run_model(**config)
            else:
                y_pred, y_true = utils.run_model(**config)
            if args.cv:
                oof_preds[valid_index] = y_pred
        if args.cv:
            valid_preds = oof_preds
            valid_true = train_df['diagnosis']
        else:
            valid_preds = y_pred
            valid_true = y_true
        if task == 'class':
            round_valid_preds = np.argmax(valid_preds, axis=1)
        elif task == 'reg':
            print('optimizing threshold ...')
            optR = utils.OptimizedRounder()
            optR.fit(valid_preds, valid_true)
            coef = optR.coefficients()
            print(f'best coef: {coef}')
            if azure_run:
                azure_run.log('coef', coef)
            round_valid_preds = optR.predict(valid_preds, coef)
        val_kappa = cohen_kappa_score(round_valid_preds, valid_true,
                                      weights='quadratic')
            
        print(f'best val kappa: {val_kappa}')
        if azure_run:
            azure_run.log('best val kappa', val_kappa)

        test_csv = pd.read_csv(utils.TEST_CSV_PATH)
        #test_tfms = utils.build_transform(size=IMAGE_SIZE, mode='test')
        test_tfms = utils.build_transform(size=IMAGE_SIZE, mode='val')
        test_dataset = RetinopathyDataset(df=test_csv, mode='test', transform=test_tfms,
                                          auto_crop=True, add_blur=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                                  shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)

        test_preds = np.zeros((len(test_csv), n_class))
        for i in range(len(indices)):
            model = utils.load_pytorch_model(model_name, result_dir / f'model_fold{i}', n_class)
            test_preds += utils.predict(model, test_loader, n_class=n_class, device=device, tta=1)
        test_preds /= len(indices)
        if task == 'class':
            round_test_preds = np.argmax(test_preds, axis=1)
        elif task == 'reg':
            round_test_preds = optR.predict(test_preds, coef)
        submission_csv = pd.read_csv(utils.SAMPLE_SUBMISSION_PATH)
        submission_csv['diagnosis'] = round_test_preds
        submission_csv.to_csv(result_dir / 'submission.csv', index=False)
        
        print('finish!!!')
        if not args.debug:
            slack.notify_finish(experiment_name, config, val_kappa)
        
    except KeyboardInterrupt as e:
        if not args.debug:
            slack.notify_fail(experiment_name, config, 
                              e.__class__.__name__, str(e))
    except Exception as e:
        if azure_run:
            azure_run.fail(e)
        if not args.debug:
            slack.notify_fail(experiment_name, config, 
                              e.__class__.__name__, str(e))
        raise
    finally:
        if azure_run:
            azure_run.complete()
            print('close azure_run')
        if tb_writer:
            tb_writer.export_scalars_to_json(os.path.join(result_dir, 
                                                       'all_scalars.json'))
            tb_writer.close()
            print('close tb_writer')
        
    
if __name__ == "__main__":
    main()
    
    
    
    
    