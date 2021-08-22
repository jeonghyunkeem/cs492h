import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import quaternion
import torch
from torch.cuda import init
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
RET_DIR = os.path.join(ROOT_DIR, 'models/retrieval/dump')
from models.retrieval.autoencoder import PointNetAE

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
sys.path.append(os.path.join(ROOT_DIR, 'scan2cad'))
from s2c_ret import Scan2CADDataset
from s2c_config import Scan2CADDatasetConfig

import s2c_utils
import pickle

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device) # change allocation of current GPU
print(device)

# Hyperparameters
BATCH_SIZE = 8
NUM_POINT = 20000

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

TRAIN_DATASET = Scan2CADDataset('train', num_points=NUM_POINT, augment=False)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)
DC = Scan2CADDatasetConfig()

# Init Model
net = PointNetAE(latent=512, in_dims=3, n_points=2048)
net.load_state_dict(torch.load(RET_DIR + '/model512_50.pth'))
net.cuda()

# Load pickle
DATA_CAT = np.load(RET_DIR + '/all_category.npy')
DATA_FILENAMES = np.load(RET_DIR + '/all_filenames.npy')

# Convert Data type from cat_id(string) to class(int)
DATA_CLASS = np.zeros((DATA_CAT.shape[0]), dtype=np.int64)
for i in range(len(DATA_CAT)):
    cat_id = DATA_CAT[i][0]
    if cat_id in DC.ShapenetIDToName:
        cat_name = DC.ShapenetIDToName[cat_id]
        DATA_CAT[i] = cat_name
        DATA_CLASS[i] = int(DC.ShapenetNameToClass[cat_name])
    else:
        DATA_CAT[i] = 'other'
        DATA_CLASS[i] = int(DC.ShapenetNameToClass['other'])
ALL_CAT = np.unique(DATA_CAT)
ALL_CLASS = np.unique(DATA_CLASS)
print(ALL_CAT, ALL_CLASS)

# ================================================================================================================================================
def visualize():
    return 0

# ================================================================================================================================================
ShapenetNameToClass = {'chair': 0, 'table': 1, 'cabinet': 2, 'trash bin': 3, 'bookshelf': 4,'display': 5,'sofa': 6, 'bathtub': 7, 'other': 8}
ShapenetClassToName = {ShapenetNameToClass[k]: k for k in ShapenetNameToClass}

def get_top_8_category(cat_id):
    if cat_id > 7:
        cat = 8
    else:
        cat = cat_id

    return cat

def init_dict():
    ret_dict = {}
    ret_dict['n_total'] = 0
    ret_dict['n_good'] = 0
    ret_dict['mean_acc'] = 0.0
    ret_dict['gt_cls'] = {}
    ret_dict['pred_cls'] = {}

    ret_dict['gt_cls']      = {ALL_CLASS[k]: 0 for k in range(len(ALL_CLASS))}
    ret_dict['pred_cls']    = {ALL_CLASS[k]: 0 for k in range(len(ALL_CLASS))} 
    
    return ret_dict

# ================================================================================================================================================
def retrieval(points, database):
    with torch.no_grad():
        embedding = net(points, r=True) # (B, 512)
    
    embedding = embedding.detach().cpu()
    dist, pred_idx = database.query(embedding, k=1) # (B, 1)
 
    pred_sem_cls   = DATA_CLASS[pred_idx]         # (B, 1)
    pred_filename  = DATA_FILENAMES[pred_idx].squeeze(-1)   # (B, 1)

    # pred_sem_cls = DC.ShapenetNameToClass[pred_sem_cls]
    return pred_sem_cls, pred_filename

def ret_test():
    ret_dict = init_dict()
    net.eval()
    with open(RET_DIR + '/shapenet_kdtree.pickle', 'rb') as pickle_file:
        database_kdtree = pickle.load(pickle_file)
        for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
            if batch_idx % 10 == 0:
                print('Batch: %d'%(batch_idx))
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)
            
            # Load labels
            ins_verts = batch_data_label['ins_verts']   # (B, K, 2048, 3)
            B = ins_verts.shape[0]
            K = ins_verts.shape[1]
            gt_sem_clses = batch_data_label['sem_cls_label']    # (B, K, 1)

            B_n_total = batch_data_label['n_total'].sum().item()   # (B, )
            ret_dict['n_total'] += B_n_total
            # Collect Retrievals
            
            for ins in range(K):
                cad_pc = ins_verts[:, ins]  # (B, 2048, 3)
                if np.sum(cad_pc.cpu().numpy()) == 0:
                    continue
                batch_pred_sem_cls, batch_pred_filename = retrieval(cad_pc, database_kdtree)

                batch_pred_sem_cls = torch.from_numpy(batch_pred_sem_cls).to(device).squeeze(-1)
                batch_gt_cls = gt_sem_clses[:, ins]
                is_same_class = batch_gt_cls == batch_pred_sem_cls # (B, 1)
                # Update dictionary
                for i, i_cls in enumerate(batch_gt_cls):
                    gt_cls = i_cls.item()
                    ret_dict['gt_cls'][gt_cls] += 1
                # Good prediction
                if torch.sum(is_same_class) > 0:
                    ret_dict['n_good'] += torch.sum(is_same_class).item()
                    # Update dictionary
                    for i, ic in enumerate(is_same_class):
                        if ic:
                            pred_cls = batch_pred_sem_cls[i].item()
                            ret_dict['pred_cls'][pred_cls] += 1
                    
            if batch_idx % 10 == 0:
                mean_acc = ret_dict['n_good']/ret_dict['n_total']
                print(mean_acc)

    ret_dict['mean_acc'] = ret_dict['n_good']/ret_dict['n_total']
    for i, key in enumerate(ret_dict.keys()):
        print(key, ret_dict[key])
                
if __name__=='__main__':
    ret_test()
