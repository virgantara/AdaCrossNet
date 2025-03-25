#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import torch
import h5py
import numpy as np
import random
from torch.utils.data import Dataset

def read_off(file):
    with open(file, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise 'Not a valid OFF header'
        n_verts, n_faces, _ = tuple(map(int, f.readline().strip().split(' ')))
        verts = [list(map(float, f.readline().strip().split(' '))) for _ in range(n_verts)]
        verts = np.array(verts)
        return verts


class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data_paths = []

        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name, split)
            for file in os.listdir(class_folder):
                if file.endswith('.off'):
                    self.data_paths.append((os.path.join(class_folder, file), label))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path, label = self.data_paths[idx]
        point_cloud = read_off(file_path)

        # Random sampling if more points than num_points
        if point_cloud.shape[0] >= self.num_points:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)

        point_cloud = point_cloud[choice, :]

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return torch.tensor(point_cloud, dtype=torch.float32), label

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def load_data(partition):
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # print('DATA_DIR=>',DATA_DIR)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', normalize=False):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Subset(Dataset):
    def __init__(self, num_points, partition='train', normalize=False, percent=1.0):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        self.percent = percent

        if self.percent < 1:
            self.data, self.label = self.sample_data()
            print("Remain data and label:", self.data.shape, self.label.shape)

    def sample_data(self):
        data_by_label = {}
        for i in range(len(self.data)):
            label = self.label[i][0]
            if label not in data_by_label:
                data_by_label[label] = [self.data[i]]
            else:
                data_by_label[label].append(self.data[i])
        chosen_data = []
        chosen_label = []
        all_data = []
        all_label = []
        for label in data_by_label:
            idx = list(range(len(data_by_label[label])))
            cidx = np.random.choice(idx)
            chosen_data.append(data_by_label[label][cidx])
            chosen_label.append(label)
            del data_by_label[label][cidx]
            all_data.extend(data_by_label[label])
            all_label.extend([label]*len(data_by_label[label]))
        remain_num = int(round(len(self.data) * self.percent)) - len(chosen_data)
        idx = list(range(len(all_data)))
        cidx = random.sample(idx, remain_num)
        chosen_data = np.array(chosen_data)
        chosen_label = np.array(chosen_label)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        chosen_data = np.concatenate([chosen_data, all_data[cidx]], 0)
        chosen_label = np.concatenate([chosen_label, all_label[cidx]], 0)
        chosen_label = chosen_label.reshape(-1, 1)
        return chosen_data, chosen_label

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = ModelNet40Subset(1024, percent=0.01)
    test = ModelNet40Subset(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
