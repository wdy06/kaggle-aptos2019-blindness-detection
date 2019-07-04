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



import utils
from dataset import RetinopathyDataset

parser = argparse.ArgumentParser(description='aptos2019 blindness detection on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--no_cache", help="extract feature without cache",
                    action="store_true")
args = parser.parse_args()

def main():
    EPOCHS = 10
    BATCH_SIZE = 256
    
    if args.debug:
        print('running in debug mode')
        EPOCHS = 2
    if args.debug:
        result_dir = os.path.join(utils.RESULT_DIR, 'debug-'+datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
    else:
        result_dir = os.path.join(utils.RESULT_DIR, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
    os.mkdir(result_dir)
    print(f'created: {result_dir}')
    
    device = torch.device("cuda:0")
    utils.run_model(epochs=EPOCHS, batch_size=BATCH_SIZE, device=device, 
                    result_dir=result_dir, debug=args.debug)

    model = utils.load_pytorch_model('resnet34', os.path.join(result_dir, 'best_model'))

    test_csv = pd.read_csv(utils.TEST_CSV_PATH)
    test_dataset = RetinopathyDataset(df=test_csv, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                              shuffle=False, pin_memory=True)

    test_preds = utils.predict(model, test_loader, n_class=5, device=device)
    submission_csv = pd.read_csv(utils.SAMPLE_SUBMISSION_PATH)
    submission_csv['diagnosis'] = np.argmax(test_preds, axis=1)
    submission_csv.to_csv(os.path.join(result_dir, 'submission.csv'),
                          index=False)
    print('finish!!!')
    
if __name__ == "__main__":
    main()
    
    
    
    
    