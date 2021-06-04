#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data

# import nyuName2ID as nyu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RET_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(RET_DIR)
ROOT2_DIR = os.path.dirname(ROOT_DIR)
HOME_DIR = os.path.dirname(ROOT2_DIR)

DATA_PATH = os.path.join(HOME_DIR, 'Dataset/ShapeNet')

shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

CARED_CATEGORY = {'03001627': 'chair',
                    '04379243': 'table',
                    '02933112': 'cabinet',
                    '02747177': 'trash bin',
                    '02871439': 'bookshelf',
                    '03211117': 'display',
                    '04256520': 'sofa',
                    '02808440': 'bathtub',
                    "02818832": 'bed',
                    "03337140": 'file cabinet',
                    "02773838": 'bag',
                    "04004475": 'printer',
                    "04554684": 'washer',
                    "03636649": 'lamp',
                    "03761084": 'microwave',
                    "04330267": 'stove',
                    "02801938": 'basket',
                    "02828884": 'bench',
                    "03642806": 'laptop',
                    "03085013": 'keyboard'}

NAME2CLASS = {CARED_CATEGORY[key]:i for i, (key, item) in enumerate(CARED_CATEGORY.items())}
NAME2CLASS['other'] = 20

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(self, dataset_name='shapenetcorev2',
                class_choice=None, num_points=2048, split='train', 
                load_name=True, load_file=True,
                segmentation=False, 
                random_rotate=False, random_jitter=False, random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetcorev2', 'shapenetpart'] and segmentation == True:
            raise AssertionError

        # self.root = os.path.join(root, dataset_name + '_' + '*hdf5_2048')
        self.root = os.path.join(DATA_PATH, dataset_name + '_' + 'hdf5_2048') #*hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if class_choice:
            class_choice_list = []
            for i, (key, value) in enumerate(CARED_CATEGORY.items()):
                class_choice_list.append(value)

        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)    # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = self.load_json(self.path_file_all)    # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = np.where(np.isin(self.name, class_choice_list) == True)[0]
            self.name = np.array(self.name)
            self.name = self.name[indices].tolist()
            self.data = self.data[indices]
            self.label = self.label[indices]
            # if self.segmentation:
            #     self.seg = self.seg[indices]
            #     self.seg_num_all = shapenetpart_seg_num[id_choice]
            #     self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = np.array(self.file)
                self.file = self.file[indices].tolist()
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json'%type)
            self.path_file_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
            name = NAME2CLASS[name]
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label, seg, name, file
        else:
            return point_set, label, name, file

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    root = os.getcwd()

    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'shapenetcorev2'
    batch_size = 32

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    split = 'train' # 'test', 'all', 'val'
    data = Dataset(dataset_name='shapenetcorev2', num_points=2048, split=split, class_choice=True)
    print("datasize:", data.__len__())
    
    n = len(data)
    for i in range(n):
        ps, lb, cat, fn = data[i]
        if cat not in NAME2CLASS:
            print(cat, fn)
            break

    # item = 0
    # ps, lb, n, f = data[item]
    # print(ps.size(), ps.type(), lb, lb.size(), lb.type(), n, f) 
    # item = 1
    # ps, lb, n, f = data[item]
    # print(ps.size(), ps.type(), lb, lb.size(), lb.type(), n, f) 
    # item = 2
    # ps, lb, n, f = data[item]
    # print(ps.size(), ps.type(), lb, lb.size(), lb.type(), n, f) 