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

import math
import cv2
import os
import sys
import time
import shutil
from itertools import chain
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import skimage.transform
from peterpy import peter
from ballpark import ballpark

import losses
from models import unet_model
from metrics import Judge
import utils
import data
from data import csv_collator
from data import RandomHorizontalFlipImageAndLabel
from data import RandomVerticalFlipImageAndLabel
from data import ScaleImageAndLabel
from collections import Counter


seed_test = 0
device_cpu = torch.device('cpu')
device = torch.device('cuda')
np.random.seed(seed_test)
torch.manual_seed(seed_test)
test_height=256
test_width=256
test_p = -1
num_epochs = 10000000
max_trainset_size =1
n_classes = 16
lambdaa = 1
log_interval = 3
val_dir = None
val_freq = 1
save_path = '/home/devs/570/zproject/locating-objects-classes/object-locator/checkpoints/'
save_file = 'final_test_16classes_ultrasmall_one_image'
n_points = None
radius = 5
paint = True
max_mask_pts = np.infty
lr = 1e-3
best_avg = 50

trainset_loader, valset_loader = \
    data.get_train_val_loaders(train_dir="/home/devs/570/zproject/locating-objects-classes/mpii_human_pose_v1/images/",
                               max_trainset_size=max_trainset_size,
                               collate_fn=csv_collator,
                               height=test_height,
                               width=test_width,
                               seed = seed_test,
                               val_dir=val_dir,
                               batch_size=1)
    


with peter('Building network'):
    model = unet_model.UNet(3, n_classes=n_classes,
                            height=test_height,
                            width=test_width,
                            known_n_points=None,
                            device=device,
                            ultrasmall=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" with {ballpark(num_params)} trainable parameters. ", end='')
model = nn.DataParallel(model)
model.to(device)

loss_regress = nn.SmoothL1Loss()
loss_loc = losses.WeightedHausdorffDistance(resized_height=test_height,
                                            resized_width=test_width,
                                            p=test_p,
                                            return_2_terms=True,
                                            device=device)




optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       amsgrad=True)


start_epoch = 0
lowest_mahd = np.infty

# Restore saved checkpoint (model weights + epoch + optimizer state)
#if args.resume:
#    with peter('Loading checkpoint'):
#        if os.path.isfile(args.resume):
#            checkpoint = torch.load(args.resume)
#            start_epoch = checkpoint['epoch']
#            try:
#                lowest_mahd = checkpoint['mahd']
#            except KeyError:
#                lowest_mahd = np.infty
#                print('W: Loaded checkpoint has not been validated. ', end='')
#            model.load_state_dict(checkpoint['model'])
#            if not args.replace_optimizer:
#                optimizer.load_state_dict(checkpoint['optimizer'])
#            print(f"\n\__ loaded checkpoint '{args.resume}'"
#                  f"(now on epoch {checkpoint['epoch']})")
#        else:
#            print(f"\n\__ E: no checkpoint found at '{args.resume}'")
#            exit(-1)

running_avg = utils.RunningAverage(len(trainset_loader))

normalzr = utils.Normalizer(test_height, test_width)

# Time at the last evaluation
tic_train = -np.infty
tic_val = -np.infty

epoch = start_epoch
it_num = 0

### test epoch trainer

while epoch < num_epochs:

    loss_avg_this_epoch = 0
    iter_train = tqdm(trainset_loader,
                      desc=f'Epoch {epoch} ({len(trainset_loader.dataset)} images)')

    # === TRAIN ===
    # Set the module in training mode
    model.train()

    for batch_idx, (imgs, dictionaries) in enumerate(iter_train):
        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_counts = [dictt['count'].to(device)
                         for dictt in dictionaries]
        target_classes = [dictt['classes'].to(device)
                            for dictt in dictionaries]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dictionaries]

        # Lists -> Tensor batches
        target_counts = torch.stack(target_counts)
        target_orig_heights = torch.stack(target_orig_heights)
        target_orig_widths = torch.stack(target_orig_widths)
        target_orig_sizes = torch.stack((target_orig_heights,
                                         target_orig_widths)).transpose(0, 1)

        # One training step
        ###########HEREEEEEEEEEEEE
        optimizer.zero_grad()
        est_maps, est_counts = model.forward(imgs)
        one_loss = []

        for class_id, (one_map, one_count) in enumerate(zip(est_maps, est_counts)):
            one_map = one_map.squeeze(1)
            
            final_target_locations = []
            if class_id in target_classes[0].tolist():
                target_counts = torch.tensor([target_classes[0].tolist().count(class_id)], dtype=torch.get_default_dtype())
                indices = [i for i, x in enumerate(target_classes[0].tolist()) if x == class_id]
                for idx in indices:
                    final_target_locations.append(target_locations[0].tolist()[idx])
                final_target_locations = torch.tensor(final_target_locations, dtype=torch.get_default_dtype())
            else:
                target_counts = torch.tensor([0], dtype=torch.get_default_dtype())
                final_target_locations = torch.tensor([-1, -1], dtype=torch.get_default_dtype())
            
            target_counts = torch.stack([target_counts.to(device)])
            final_target_locations = [final_target_locations.to(device)]
            
            
            term1, term2 = loss_loc.forward(one_map,
                                            final_target_locations,
                                            target_orig_sizes)
            one_count = one_count.view(-1)
            target_counts = target_counts.view(-1)
            term3 = loss_regress.forward(one_count, target_counts)
            term3 *= lambdaa
            one_loss.append(term1 + term2 + term3)
        
        loss = sum(one_loss)
        loss.backward()
        optimizer.step()

        # Update progress bar
        running_avg.put(loss.item())
        iter_train.set_postfix(running_avg=f'{round(running_avg.avg/3, 1)}')

        # Log training error
        if time.time() > tic_train + log_interval:
            tic_train = time.time()
            orig_shape = target_orig_sizes[0].data.to(device_cpu).numpy().tolist()
            orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().to(device_cpu).numpy().transpose((1, 2, 0)),
                                                       output_shape=orig_shape,
                                                       mode='constant') + 1) / 2.0 * 255.0).\
            astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(one_map[0].data.unsqueeze(0).to(device_cpu).numpy().transpose((1, 2, 0)),
                                                    output_shape=orig_shape,
                                                    mode='constant').\
            astype(np.float32).transpose((2, 0, 1)).squeeze(0)

            # Overlay output on heatmap
            orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                map=est_map_origsize).\
                astype(np.float32)

            # Send heatmap with circles at the labeled points to Visdom
            target_locs_np = target_locations[0].\
                to(device_cpu).numpy().reshape(-1, 2)
            target_orig_size_np = target_orig_sizes[0].\
                to(device_cpu).numpy().reshape(2)
            target_locs_wrt_orig = normalzr.unnormalize(target_locs_np,
                                                        orig_img_size=target_orig_size_np)
            img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                               points=target_locs_wrt_orig,
                                               color='white')
        it_num += 1

    # Never do validation?
    if not val_dir or \
            not valset_loader or \
            len(valset_loader) == 0 or \
            val_freq == 0:
        
        # Time to save checkpoint?
        if save_file and (epoch + 1) % val_freq == 0:
            
            if best_avg > round(running_avg.avg/3, 1):
                print("saving best MODEL!!! Good job!! Woohooo")
                best_avg = round(running_avg.avg/3, 1)
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'n_points': n_points,
                            }, (save_path+save_file+ 'running_avg_'+ str(best_avg)+'.ckpt'))
        epoch += 1
        continue

    # Time to do validation?
    if (epoch + 1) % val_freq != 0:
        epoch += 1
        continue

    # === VALIDATION ===

    # Set the module in evaluation mode
    model.eval()

    judge = Judge(r=radius)
    sum_term1 = 0
    sum_term2 = 0
    sum_term3 = 0
    sum_loss = 0
    iter_val = tqdm(valset_loader,
                    desc=f'Validating Epoch {epoch} ({len(valset_loader.dataset)} images)')
    for batch_idx, (imgs, dictionaries) in enumerate(iter_val):

        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_counts = [dictt['count'].to(device)
                        for dictt in dictionaries]
        target_classes = [dictt['classes'].to(device)
                            for dictt in dictionaries]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dictionaries]

        with torch.no_grad():
            target_counts = torch.stack(target_counts)
            target_orig_heights = torch.stack(target_orig_heights)
            target_orig_widths = torch.stack(target_orig_widths)
            target_orig_sizes = torch.stack((target_orig_heights,
                                             target_orig_widths)).transpose(0, 1)
        orig_shape = (dictionaries[0]['orig_height'].item(),
                      dictionaries[0]['orig_width'].item())
        # Feed-forward
        with torch.no_grad():
            est_maps, est_counts = model.forward(imgs)
        
        for class_id in range(n_classes):
            for one_map, one_count in zip(est_maps, est_counts):
                one_map = one_map.squeeze(1)
                final_target_locations = []
                if class_id in target_classes[0].tolist():
                    target_counts = torch.tensor(target_classes[0].tolist().count(class_id), dtype=torch.get_default_dtype())
                    indices = [i for i, x in enumerate(target_classes[0].tolist()) if x == class_id]
                    for idx in indices:
                        final_target_locations.append(target_locations[0].tolist()[idx])
                    final_target_locations = torch.tensor(final_target_locations, dtype=torch.get_default_dtype())
                else:
                    target_counts = torch.tensor([0], dtype=torch.get_default_dtype())
                    final_target_locations = torch.tensor([-1, -1], dtype=torch.get_default_dtype())
                
                target_counts = torch.stack([target_counts.to(device)])
                final_target_locations = [final_target_locations.to(device)]
                
                # Tensor -> float & numpy
                target_count_int = int(round(target_counts.item()))
                target_locations_np = \
                    final_target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
                target_orig_size_np = \
                    target_orig_sizes[0].to(device_cpu).numpy().reshape(2)
        
                normalzr = utils.Normalizer(test_height, test_width)
        
                if target_count_int == 0:
                    continue
        
               
        
                # Tensor -> int
                one_count_int = int(round(one_count.item()))
        
                # The 3 terms
                with torch.no_grad():
                    one_count = one_count.view(-1)
                    target_counts = target_counts.view(-1)
                    term1, term2 = loss_loc.forward(one_map,
                                                    final_target_locations,
                                                    target_orig_sizes)
                    term3 = loss_regress.forward(one_count, target_counts)
                    term3 *= lambdaa
                sum_term1 += term1.item()
                sum_term2 += term2.item()
                sum_term3 += term3.item()
        sum_loss += term1 + term2 + term3
        
        # Update progress bar
        loss_avg_this_epoch = sum_loss.item() / (batch_idx + 1)
        iter_val.set_postfix(
            avg_val_loss_this_epoch=f'{loss_avg_this_epoch:.1f}-----')

        # The estimated map must be thresholed to obtain estimated points
        # BMM thresholding
        est_map_numpy = one_map[0, :, :].to(device_cpu).numpy()
        est_map_numpy_origsize = skimage.transform.resize(est_map_numpy,
                                                          output_shape=orig_shape,
                                                          mode='constant')
        mask, _ = utils.threshold(est_map_numpy_origsize, tau=-1)
        # Obtain centroids of the mask
        centroids_wrt_orig = utils.cluster(mask, one_count_int,
                                           max_mask_pts=max_mask_pts)

        # Validation metrics
        target_locations_wrt_orig = normalzr.unnormalize(target_locations_np,
                                                         orig_img_size=target_orig_size_np)
        judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                          max_ahd=loss_loc.max_dist)
        judge.feed_count(one_count_int, target_count_int)

        if time.time() > tic_val + log_interval:
            tic_val = time.time()

            # Resize to original size
            orig_img_origsize = ((skimage.transform.resize(imgs[0].to(device_cpu).squeeze().numpy().transpose((1, 2, 0)),
                                                           output_shape=target_orig_size_np.tolist(),
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(one_map[0].to(device_cpu).unsqueeze(0).numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1)).squeeze(0)

            # Overlay output on heatmap
            orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                map=est_map_origsize).\
                astype(np.float32)

            if not paint:
                pass

            else:
                # Send heatmap with a cross at the estimated centroids to Visdom
                img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                 points=centroids_wrt_orig,
                                                 color='red',
                                                 crosshair=True )


    avg_term1_val = sum_term1 / len(valset_loader)
    avg_term2_val = sum_term2 / len(valset_loader)
    avg_term3_val = sum_term3 / len(valset_loader)
    avg_loss_val = sum_loss / len(valset_loader)

    # If this is the best epoch (in terms of validation error)
    if judge.mahd < lowest_mahd:
        # Keep the best model
        lowest_mahd = judge.mahd
        if save_file:
            torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                        'model': model.state_dict(),
                        'mahd': lowest_mahd,
                        'optimizer': optimizer.state_dict(),
                        'n_points': n_points,
                        }, (save_path+save_file+ 'running_avg_'+ str(best_avg)+'.ckpt'))
            print("Saved best checkpoint so far in %s " % (save_path+save_file+ 'running_avg_'+ str(best_avg)+'.ckpt'))

    epoch += 1

"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
