# -*- coding: utf-8 -*

"""
Note: CIT-GAN is built over StarGAN v2 to improve the performance
Author of CIT-GAN: Shivangi Yadav
Advisor: Dr. Arun Ross

Reference:
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model2
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher2
import core.utils as utils
from metrics.eval import calculate_metrics

from time import sleep
from tqdm import tqdm

import pdb
from pathlib import Path


class Solver2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model2(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)

        if args.mode == 'train_style' or args.mode == 'test_style':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir + '\\' + "{:06d}_nets_prestyle.ckpt"), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir + '\\' + "{:06d}_nets_ema_prestyle.ckpt"), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir + '\\' + "{:06d}_optims_prestyle.ckpt"), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(Path(ospj(args.checkpoint_dir + '\\' + "{:06d}_nets_ema_prestyle.ckpt")), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims
        cls_loss = nn.CrossEntropyLoss()
        
        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        total_samples = len(loaders.src.dataset)
        
        for i in tqdm(range(args.resume_iter, args.total_iters)):
            # fetch images and labels
            sleep(3)
            running_loss = 0.0
            running_corrects = 0.0
            
            for j, inputs in tqdm(enumerate(loaders.src)):
                #pdb.set_trace()
                sleep(3)
                x_real, y_org = inputs
                x_real = x_real.cuda()
                y_org = y_org.cuda()
    
                # train the styling network
                #print("Training Style Network")
                self._reset_grad()
                
                s_real, o_real = nets.style_encoder(x_real, y_org)
                #pdb.set_trace()
                _, preds = torch.max(o_real, 1)
                
                loss = cls_loss(o_real, y_org)
                
                loss.backward()
                optims.style_encoder.step()
                
                # print out log info
                running_loss += loss.item() * x_real.size(0)
                running_corrects += torch.sum(preds == y_org.data)
                #print("Running Loss: ", loss.item())
                #print("Running Corrects: ", torch.sum(preds == y_org.data))
    
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            print("Loss: {}".format(epoch_loss))
            print("Accuracy: {}".format(epoch_acc))
            
            
            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
    
    def test(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        
        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start testing...')
        start_time = time.time()
        total_samples = len(loaders.val.dataset)
        nets.style_encoder.eval()
        
        with torch.no_grad():
            running_loss = 0.0
            running_corrects = 0.0
            
            for j, inputs in tqdm(enumerate(loaders.val)):
                #pdb.set_trace()
                sleep(3)
                x_real, y_org = inputs
                x_real = x_real.cuda()
                y_org = y_org.cuda()
                
                s_real, o_real = nets.style_encoder(x_real, y_org)
                #pdb.set_trace()
                _, preds = torch.max(o_real, 1)
                
                
                # print out log info
                running_corrects += torch.sum(preds == y_org.data)
                #print("Running Loss: ", loss.item())
                #print("Running Corrects: ", torch.sum(preds == y_org.data))
                
        epoch_acc = running_corrects.double() / total_samples
        print("Accuracy: {}".format(epoch_acc))
        
