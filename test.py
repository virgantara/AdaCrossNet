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

from datasets.data import ShapeNetRender, ModelNet40SVM, ScanObjectNNSVM, ScanObjectNNDataset
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
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        help='Dataset name')
    return parser.parse_args()

def test(args, io):
    
    #Try to load models
    if args.model == 'dgcnn_cls':
        point_model = DGCNN_cls(args).to(device)
    elif args.model == 'dgcnn_seg':
        point_model = DGCNN_partseg(args).to(device)
    elif args.model == 'pointnet_cls':
        point_model = PointNet_cls(args).to(device)
    elif args.model == 'pointnet_seg':
        point_model = PointNet_partseg(args).to(device)
    else:
        raise Exception("Not implemented")
    
    point_model.load_state_dict(torch.load(args.model_path, weights_only=True))
    print("Model Loaded !!")
        
    
    best_acc = 0

    # Testing
    if args.dataset == 'modelnet40':
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_points), batch_size=args.batch_size, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_points), batch_size=args.test_batch_size, shuffle=False)
    elif args.dataset == 'scanobjectnn':
        train_val_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_points), batch_size=args.batch_size, shuffle=True)
        test_val_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=args.num_points), batch_size=args.test_batch_size, shuffle=False)

    feats_train = []
    labels_train = []
    point_model.eval()

    for (data, label) in tqdm(train_val_loader):
        # print('data=>',data.shape,'label=>',label.shape) #[B,num_points,3]
        labels = list(map(lambda x: x[0],label.numpy().tolist()))
        # print('labels=>',labels) #[B,1]的label标签转换为一个大小为B的数组
        data = data.permute(0, 2, 1).to(device)
        with torch.no_grad():
            feats = point_model(data)[-1]
        feats = feats.detach().cpu().numpy()
        # print('feats=>', feats.shape) #[B,2048(max1024+avg1024)]
        for feat in feats:
            feats_train.append(feat)
        labels_train += labels

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)
    # print('feats_train=>',feats_train.shape,'labels_train=>',labels_train.shape) #(9840, 2048),(9840,)

    feats_test = []
    labels_test = []

    for data, label in tqdm(test_val_loader):
        labels = list(map(lambda x: x[0],label.numpy().tolist()))
        data = data.permute(0, 2, 1).to(device)
        with torch.no_grad():
            feats = point_model(data)[-1]
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_test.append(feat)
        labels_test += labels

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
    
    model_tl = SVC(C = 0.01, kernel ='linear')

    model_tl.fit(feats_train, labels_train)
    # print('model_tl=>', model_tl)
    test_accuracy = model_tl.score(feats_test, labels_test)
    
    io.cprint(f"Linear Accuracy : {test_accuracy}")
    
    if test_accuracy > best_acc:
        best_acc = test_accuracy
            
    

if __name__ == "__main__":
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)

    test(args, io)


