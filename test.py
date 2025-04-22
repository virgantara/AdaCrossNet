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
    parser.add_argument('--dataset_name', type=str, default='modelnet40svm', metavar='N',
                        choices=['modelnet40svm', 'scanobjectnnsvm'],
                        help='Dataset name to test, [modelnet40svm, scanobjectnnsvm]')
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
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        help='Dataset name')
    return parser.parse_args()

def test(args, io):
   
    # Load test dataset
    test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_points), batch_size=args.test_batch_size, shuffle=False)

    # Load model
    if args.model == 'dgcnn_cls':
        point_model = DGCNN_cls(args).to(device)
    elif args.model == 'dgcnn_seg':
        point_model = DGCNN_partseg(args).to(device)
    elif args.model == 'pointnet_cls':
        point_model = PointNet_cls(args).to(device)
    elif args.model == 'pointnet_seg':
        point_model = PointNet_partseg(args).to(device)
    else:
        raise Exception("Model not implemented")

    point_model.load_state_dict(torch.load(args.model_path))
    point_model.eval()

    criterion = NTXentLoss(temperature=0.2).to(device)

    # Linear evaluation
    feats_test = []
    labels_test = []

    with torch.no_grad():
        for data, label in tqdm(test_val_loader, desc='Extracting Features'):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            feats = point_model(data)[-1]
            feats_test.extend(feats.cpu().numpy())
            labels_test += labels

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)

    # (Optional) create mock training set for SVM training
    # You may want to load a real one, depending on test protocol
    if args.dataset_name == 'modelnet40svm':
        train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_points), batch_size=args.batch_size, shuffle=True)
    elif args.dataset_name == 'scanobjectnnsvm':
        train_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_points), batch_size=args.batch_size, shuffle=True)
    feats_train, labels_train = [], []

    with torch.no_grad():
        for data, label in tqdm(train_loader, desc='Preparing SVM Train Set'):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            feats = point_model(data)[-1]
            feats_train.extend(feats.cpu().numpy())
            labels_train += labels

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)

    model_tl = SVC(C=0.01, kernel='linear')
    model_tl.fit(feats_train, labels_train)
    test_accuracy = model_tl.score(feats_test, labels_test)

    io.cprint(f"Test Linear Accuracy: {test_accuracy}")
    # Optional: compute test contrastive loss (e.g., between first and second halves)
    test_loss_meter = AverageMeter()
    with torch.no_grad():
        for data, _ in tqdm(test_val_loader, desc="Calculating Contrastive Loss"):
            data = data.permute(0, 2, 1).to(device)
            feats, _ = point_model(data)
            if feats.size(0) >= 2:
                split = feats.size(0) // 2
                loss = criterion(feats[:split], feats[split:2*split])
                test_loss_meter.update(loss.item(), split)

    io.cprint(f"Test Contrastive Loss: {test_loss_meter.avg:.6f}")

            
    

if __name__ == "__main__":
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)

    test(args, io)


