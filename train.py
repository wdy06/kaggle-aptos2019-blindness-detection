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

from azureml.core import Workspace, Experiment
from torch.utils.tensorboard import SummaryWriter


import utils
import slack
from dataset import RetinopathyDataset

parser = argparse.ArgumentParser(description='aptos2019 blindness detection on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--no_cache", help="extract feature without cache",
                    action="store_true")
parser.add_argument("--cv", help="do cross validation",
                    action="store_true")
parser.add_argument("--multi", help="use multi gpu",
                    action="store_true")
parser.add_argument('--model', '-m', type=str, default='resnet34',
                    help='cnn model name')
parser.add_argument('--batch', '-B', type=int, default=64,
                    help='batch size')
parser.add_argument('--size', '-S', type=int, default=256,
                    help='image size')
args = parser.parse_args()

def main():
    EPOCHS = 50
    N_FOLDS = 5
    BATCH_SIZE = args.batch
    IMAGE_SIZE = args.size
    model_name = args.model
    optimizer_name = 'adam'
    loss_name = 'crossentropy'
    lr = 0.001
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
            result_dir = os.path.join(utils.RESULT_DIR, 'debug-' + experiment_name)
        else:
            result_dir = os.path.join(utils.RESULT_DIR, experiment_name)
            ws = Workspace.from_config('.aml_config/config.json')
            exp = Experiment(workspace=ws, name='kaggle-aptos2019')
            azure_run = exp.start_logging()
            azure_run.log('experiment name', experiment_name)
            azure_run.log('epoch', EPOCHS)
            azure_run.log('batch size', BATCH_SIZE)
            azure_run.log('image size', IMAGE_SIZE)
            azure_run.log('model', model_name)
            azure_run.log('optimizer', optimizer_name)
            azure_run.log('loss_name', loss_name)
            azure_run.log('lr', lr)
            if args.cv:
                azure_run.log('cv', N_FOLDS)
            else:
                azure_run.log('cv', 0)
        if args.multi:
            print('use multi gpu !!')
            

        os.mkdir(result_dir)
        print(f'created: {result_dir}')
        
#         if not args.debug:
#             tb_writer = SummaryWriter(log_dir=result_dir)
            

        device = torch.device("cuda:0")
        config = {
                  'epochs': EPOCHS,
                  #'n_folds': N_FOLDS,
                  'multi': args.multi,
                  'batch_size': BATCH_SIZE,
                  'image_size': IMAGE_SIZE,
                  'model_name': model_name,
                  'optimizer_name': optimizer_name,
                  'loss_name': loss_name,
                  'lr': lr, 
                  'device': device,
                  #'result_dir': result_dir,
                  #'debug': args.debug,
                  'num_workers': num_workers,
                  #'azure_run': azure_run,
                  #'writer': tb_writer
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
        oof_preds = np.zeros((len(train_df), utils.N_CLASS))
        for i_fold, (train_index, valid_index) in tqdm(enumerate(indices)):
            model_path = os.path.join(result_dir, f'model_fold{i_fold}')
            config['train_index'] = train_index
            config['valid_index'] = valid_index
            config['model_path'] = model_path
            if azure_run:
                if i_fold == 0:
                    config['azure_run'] = azure_run
                    y_pred, y_true = utils.run_model(**config)
                else:
                    with azure_run.child_run() as child:
                        conifg['azure_run'] = child
                        y_pred, y_true = utils.run_model(**config)
            else:
                y_pred, y_true = utils.run_model(**config)
            if args.cv:
                oof_preds[valid_index] = y_pred
        #oof_preds, y_true = utils.run_model(**config)
        if args.cv:
            val_kappa = cohen_kappa_score(np.argmax(oof_preds, axis=1), train_df['diagnosis'])
        else:
            val_kappa = cohen_kappa_score(np.argmax(y_pred, axis=1), y_true)
            
        print(f'best val kappa: {val_kappa}')
        if azure_run:
            azure_run.log('best val kappa', val_kappa)

        test_csv = pd.read_csv(utils.TEST_CSV_PATH)
        test_tfms = utils.build_transform(size=IMAGE_SIZE, mode='test')
        test_dataset = RetinopathyDataset(df=test_csv, mode='test', transform=test_tfms,
                                          auto_crop=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                                  shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)

        test_preds = np.zeros((len(test_csv), utils.N_CLASS))
        for i in range(len(indices)):
            model = utils.load_pytorch_model(model_name, os.path.join(result_dir, f'model_fold{i}'))
            test_preds += utils.predict(model, test_loader, n_class=5, device=device)
            
        submission_csv = pd.read_csv(utils.SAMPLE_SUBMISSION_PATH)
        submission_csv['diagnosis'] = np.argmax(test_preds, axis=1)
        submission_csv.to_csv(os.path.join(result_dir, 'submission.csv'),
                              index=False)
        
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
    
    
    
    
    