from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18, resnet101
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.data import ShapeNetRender, ModelNet40SVM
from models.dgcnn import DGCNN_cls, DGCNN_partseg
from models.pointnet import PointNet_cls, PointNet_partseg
from models.resnet import ResNet
from util import IOStream, AverageMeter


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn_cls', metavar='N',
                        choices=['dgcnn_cls', 'dgcnn_seg', 'pointnet_cls', 'pointnet_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args()

def train(args, io):
    
    train_loader = DataLoader(ShapeNetRender(), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    
        
    img_rgb_model = ResNet(resnet50(), feat_dim = 2048)
    img_rgb_model = img_rgb_model.to(device)

    img_gray_model = ResNet(resnet50(), feat_dim = 2048, type = 'gray')
    img_gray_model = img_gray_model.to(device)
        
    
        
    parameters = list(img_rgb_model.parameters()) + list(img_gray_model.parameters())
    # print(parameters)

    # if args.use_sgd:
    print("Use SGD")
    opt = optim.SGD(parameters, lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    # else:
    # print("Use Adam")
    # opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature = 0.2).to(device)
    
    best_acc = 0

    # Initialize weights and their rates of change
    lambda_intra = 1.0
    lambda_cross = 1.0
    alpha = 0.05  # Control sensitivity of adjustment
    beta = 0.95  # Smoothing factor

    prev_loss_intra = 0
    prev_loss_cross = 0
    for epoch in range(args.start_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        train_intra_losses = AverageMeter()
        train_cross_losses = AverageMeter()
        
        
        img_rgb_model.train()
        img_gray_model.train()
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for (data_t1, data_t2), (img_rgb, img_gray) in tqdm(train_loader):
            data_t1, data_t2, img_rgb, img_gray = data_t1.to(device), data_t2.to(device), img_rgb.to(device), img_gray.to(device)
            # print('data_t1=>',data_t1.shape,'data_t2=>',data_t2.shape,'imgs=>',imgs.shape)
            batch_size = data_t1.size()[0]

            opt.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            
            img_rgb_feats = img_rgb_model(img_rgb)
            img_gray_feats = img_gray_model(img_gray)

            image_feats = torch.cat([img_rgb_feats, img_gray_feats],dim = 0)

            loss_cross = criterion(image_feats)

            # Dynamic weight adjustment
            delta_cross = loss_cross.item() - prev_loss_cross

           
            lambda_cross = beta * lambda_cross + (1 - beta) * (1 / (1 + np.exp(-alpha * delta_cross)))

            # Update previous loss values
            prev_loss_cross = loss_cross.item()

            total_loss = lambda_cross * loss_cross
            total_loss.backward()
            opt.step()
            lr_scheduler.step()

            train_losses.update(total_loss.item(), batch_size)
            train_intra_losses.update(loss_intra.item(), batch_size)
            train_cross_losses.update(loss_cross.item(), batch_size)

        outstr = 'Epoch (%d), Batch(%d), loss: %.6f, intra loss: %.6f, cross loss: %.6f' % \
                 (epoch, len(train_loader), train_losses.avg, train_intra_losses.avg, train_cross_losses.avg)
        io.cprint(outstr)

      


if __name__ == "__main__":
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

