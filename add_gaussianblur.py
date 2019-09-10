import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from tqdm import tqdm, tqdm_notebook
import cv2

parser = argparse.ArgumentParser(description='aptos2019 blindness detection on kaggle')
parser.add_argument('--input-dir', type=str,
                    help='path to input directory')
parser.add_argument('--output-dir', type=str,
                    help='path to output directory')
parser.add_argument('--sigma', type=int, help='sigmaX')
args = parser.parse_args()

# input_dir = './data/train_all_crop/'
# output_dir = './data/train_all_crop_blur10/'
input_dir = args.input_dir
output_dir = args.output_dir
sigmaX = args.sigma

input_dir = pathlib.Path(input_dir)

img_list = list(input_dir.glob('*'))

# sigmaX = 10

for img_path in tqdm(img_list):
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.addWeighted(image,4, cv2.GaussianBlur(image, (0,0), sigmaX),-4, 128)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, img_path.name)
    cv2.imwrite(output_path, image)
    
    
    