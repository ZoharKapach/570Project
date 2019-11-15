from __future__ import print_function

__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import sys
import time
import shutil
from parse import parse
import math
from collections import OrderedDict
import itertools

import matplotlib
matplotlib.use('Agg')
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
from torchvision.models import inception_v3
import skimage.transform
from peterpy import peter
from ballpark import ballpark

from data import csv_collator
from data import ScaleImageAndLabel
from data import build_dataset
import losses
from models import unet_model
from metrics import Judge
from metrics import make_metric_plots
import utils


# Parse command line arguments
#args = argparser.parse_command_args('testing')

# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') #if args.cuda else device_cpu
seed = 0
height = 256
width = 256
dataset = '/home/devs/570/zproject/locating-objects-classes/mpii_human_pose_v1/images/'
model = '/home/devs/570/zproject/locating-objects-classes/object-locator/checkpoints/final_test3_16classes_ultrasmall_one_imagerunning_avg_100.0.ckpt'
out_dir = '/home/devs/570/zproject/locating-objects-classes/object-locator/final_test3_4'
evaluate = False
max_testset_size = 1
nThreads = 4
cuda = True
ultrasmallnet = True
n_points = None
taus = [0.95] #np.linspace(0, 1, 25).tolist() + [-1, -2]
radii = range(0, 15 + 1)
force = False
paint = True
max_mask_pts = np.infty
n_classes = 16
force = False
results= []
# Set seeds
np.random.seed(seed)
torch.manual_seed(seed)
#if args.cuda:
torch.cuda.manual_seed_all(seed)

# Data loading code
try:
    testset = build_dataset(dataset,
                            transforms=transforms.Compose([
                                ScaleImageAndLabel(size=(height,width)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                            ]),
                            ignore_gt=False,
                            max_dataset_size=max_testset_size)
except ValueError as e:
    print(f'E: {e}')
    exit(-1)
    
testset_loader = data.DataLoader(testset,
                                 batch_size=1,
                                 num_workers=nThreads,
                                 collate_fn=csv_collator)
# Array with [height, width] of the new size
resized_size = np.array([height, width])

# Loss function
criterion_training = losses.WeightedHausdorffDistance(resized_height=height,
                                                      resized_width=width,
                                                      return_2_terms=True,
                                                      device=device)

# Restore saved checkpoint (model weights)
with peter("Loading checkpoint"):

    # Pretrained models that come with this package
    if model == 'unet_256x256_sorghum':
        model = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'checkpoints',
                                  'unet_256x256_sorghum.ckpt')
    if os.path.isfile(model):
        if cuda:
            checkpoint = torch.load(model)
        else:
            checkpoint = torch.load(
                model, map_location=lambda storage, loc: storage)
        # Model
        if n_points is None:
            if 'n_points' not in checkpoint:
                # Model will also estimate # of points

                model = unet_model.UNet(3, n_classes,
                                        known_n_points=None,
                                        height=height,
                                        width=width,
                                        ultrasmall=ultrasmallnet)

            else:
                # The checkpoint tells us the # of points to estimate
                model = unet_model.UNet(3, n_classes,
                                        known_n_points=checkpoint['n_points'],
                                        height=height,
                                        width=width,
                                        ultrasmall=ultrasmallnet)
        else:
            # The user tells us the # of points to estimate
            model = unet_model.UNet(3, n_classes,
                                    known_n_points=n_points,
                                    height=height,
                                    width=width,
                                    ultrasmall=ultrasmallnet)

        # Parallelize
        if cuda:
            model = nn.DataParallel(model)
        model = model.to(device)

        # Load model in checkpoint
        if cuda:
            state_dict = checkpoint['model']
        else:
            # remove 'module.' of DataParallel
            state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]
                state_dict[name] = v
        model.load_state_dict(state_dict)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#        print(f"\n\__ loaded checkpoint '{model}' "
#              f"with {ballpark(num_params)} trainable parameters")
        # print(model)
    else:
        print(f"\n\__  E: no checkpoint found at '{model}'")
        exit(-1)

    tic = time.time()


# Set the module in evaluation mode
model.eval()

# Accumulative histogram of estimated maps
bmm_tracker = utils.AccBetaMixtureModel()


if testset.there_is_gt:
    # Prepare Judges that will compute P/R as fct of r and th
    judges = []
    for r, th in itertools.product(radii, taus):
        judge = Judge(r=r)
        judge.th = th
        judges.append(judge)

# Empty output CSV (one per threshold)
df_outs = [pd.DataFrame() for _ in taus]

# --force will overwrite output directory
if force:
    shutil.rmtree(out_dir)

myImage = cv2.imread(dataset + '033324174.jpg')
myImage = cv2.resize(myImage, (256, 256))
myPoints = []        
for batch_idx, (imgs, dictionaries) in tqdm(enumerate(testset_loader),
                                            total=len(testset_loader)):
    # Move to device
    imgs = imgs.to(device)

    # Pull info from this batch and move to device
    if testset.there_is_gt:
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_count = [dictt['count'].to(device)
                        for dictt in dictionaries]
        target_classes = [dictt['classes'].to(device)
                            for dictt in dictionaries]

    target_orig_heights = [dictt['orig_height'].to(device)
                           for dictt in dictionaries]
    target_orig_widths = [dictt['orig_width'].to(device)
                          for dictt in dictionaries]

    # Lists -> Tensor batches
    if testset.there_is_gt:
        target_count = torch.stack(target_count)
    target_orig_heights = torch.stack(target_orig_heights)
    target_orig_widths = torch.stack(target_orig_widths)
    target_orig_sizes = torch.stack((target_orig_heights,
                                     target_orig_widths)).transpose(0, 1)
    origsize = (dictionaries[0]['orig_height'].item(),
                dictionaries[0]['orig_width'].item())
    # Tensor -> float & numpy
    if testset.there_is_gt:
        target_count = target_count.item()
        target_locations = \
            target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
    target_orig_size = \
        target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

    normalzr = utils.Normalizer(height, width)
    
    # Feed forward
    with torch.no_grad():
        est_maps, est_count = model.forward(imgs)

    for class_id, (one_map, one_count) in enumerate(zip(est_maps, est_count)):
        one_map = one_map.squeeze(1)
        # Convert to original size
        est_map_np = one_map[0, :, :].to(device_cpu).numpy()
        est_map_np_origsize = \
            skimage.transform.resize(est_map_np,
                                     output_shape=origsize,
                                     mode='constant')
        orig_img_np = imgs[0].to(device_cpu).squeeze().numpy()


        orig_img_np_origsize = ((skimage.transform.resize(orig_img_np.transpose((1, 2, 0)),
                                                       output_shape=origsize,
                                                       mode='constant') + 1) / 2.0 * 255.0).\
            astype(np.float32).transpose((2, 0, 1))
        # Overlay output on original image as a heatmap
        orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_np_origsize,
                                                            map=est_map_np_origsize).\
            astype(np.float32)
    
        # Save estimated map to disk
        os.makedirs(os.path.join(out_dir, 'intermediate', 'estimated_map'),
                    exist_ok=True)
        cv2.imwrite(os.path.join(out_dir,
                                 'intermediate',
                                 'estimated_map',
                                 dictionaries[0]['filename']),
                    orig_img_w_heatmap_origsize.transpose((1, 2, 0))[:, :, ::-1])
    
        # Tensor -> int
        est_count_int = int(round(one_count.item()))
    
        # The estimated map must be thresholded to obtain estimated points
        for t, tau in enumerate(taus):
            if tau == 0.95:
                mask, _ = utils.threshold(est_map_np_origsize, tau)
                centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                                   max_mask_pts=max_mask_pts)
                if paint:
                    myPoints.append(centroids_wrt_orig)


# Write CSVs to disk
os.makedirs(os.path.join(out_dir, 'estimations'), exist_ok=True)
for df_out, tau in zip(df_outs, taus):
    df_out.to_csv(os.path.join(out_dir,
                               'estimations',
                               f'estimations_tau={round(tau, 4)}.csv'))

os.makedirs(os.path.join(out_dir, 'intermediate', 'metrics_plots'),
            exist_ok=True)


color = [0,0,255]
for point in myPoints:
    point = point[0]
    x, y = point[1], point[0]
    myImage = cv2.circle(myImage, (x, y), 3, color, -1)
os.makedirs(os.path.join(out_dir,'wooohooo','painted_on_original', f'tau={round(tau, 4)}'), exist_ok=True)
cv2.imwrite(os.path.join(out_dir,
                         'wooohooo',
                         'painted_on_original',
                         f'tau={round(tau, 4)}',
                         dictionaries[0]['filename']),
                         myImage)
elapsed_time = int(time.time() - tic)
print(f'It took {elapsed_time} seconds to evaluate all this dataset.')


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
