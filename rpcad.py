# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
# from train import TRAIN_DATALOADER
import numpy as np
from datetime import datetime
import argparse
import importlib
import quaternion

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler
from utils.tf_visualizer import Visualizer as TfVisualizer
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
sys.path.append(os.path.join(ROOT_DIR, 'scan2cad'))
from s2c_eval import Evaluation
from s2c_dataset import Scan2CADDataset
from s2c_config import Scan2CADDatasetConfig

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device) # change allocation of current GPU
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log2', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=64, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.002]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=50, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='200, 300, 400', help='When to decay the learning rate (in epochs) [default: 200,300,400]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint_rpcad.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_relation.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

#-----------------------------------------------------------------------------------------------------
# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
TRAIN_DATASET = Scan2CADDataset('train', num_points=NUM_POINT, augment=True)
TEST_DATASET = Scan2CADDataset('val', num_points=NUM_POINT, augment=False)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size = BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
DATASET_CONFIG = Scan2CADDatasetConfig() 

# Init the model and optimzier
MODEL = importlib.import_module('RPCADnet')
# num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1
num_input_channel = 0

# Detector
Detector = MODEL.RPCADNet
net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               input_feature_dim=num_input_channel,
               num_proposal=FLAGS.num_target,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling,
               )
if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)

# Set criterion
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Used for evaluation 
CONFIG_DICT = {'remove_empty_box':False, 
                'use_3d_nms':True,
                'nms_iou':0.25, 
                'use_old_type_nms':False, 
                'cls_nms':True,
                'per_class_proposal': True, 
                'conf_thresh':0.05,
                'dataset_config':DATASET_CONFIG}

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    evaluation = Evaluation()
    net.train() # set model to training mode
    total_loss = 0.0
    n = len(TRAIN_DATASET)
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss.backward()
        optimizer.step()
            
        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict}, 
                (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            evaluation.step(end_points)

        total_loss += loss * BATCH_SIZE
    
    class_mean_accuracy = 0

    if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
        print("\n------------------------------------------")
        class_mean_accuracy, t, r, s, eval_dict = evaluation.summary()

        for key in sorted(eval_dict.keys()):
            log_string("train {:>10s}: {:>4.4f} \t ({:>4d}/{:>4d})".format(key, eval_dict[key][0], eval_dict[key][1], eval_dict[key][2]))
            log_string("    \t (t:{:>4d}, r:{:>4d}, s:{:>4d} / {:>4d})".format(eval_dict[key][3], eval_dict[key][4], eval_dict[key][5], eval_dict[key][6]))

        print("------------------------------------------")
        log_string('train class mean center:     %f'%(t))
        log_string('train class mean rotation:   %f'%(r))
        log_string('train class mean scale:      %f'%(s))
        print("------------------------------------------")
        log_string('train class mean accuracy: %f'%(class_mean_accuracy))
        print("------------------------------------------\n")
    
    return total_loss / n, class_mean_accuracy
    

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    evaluation = Evaluation()
    net.eval() # set model to eval mode (for bn and dp)
    total_loss = 0.0
    n = len(TEST_DATASET)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        total_loss += loss * BATCH_SIZE

        # Evaluate
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            evaluation.step(end_points)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Write results
    TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict}, 
        (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)

    mean_loss = stat_dict['loss']/float(batch_idx+1)

    class_mean_accuracy = 0 
    if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
        class_mean_accuracy, t, r, s, eval_dict = evaluation.summary()

        print("\n--------- EVALUATION PER CLASS -----------")
        for key in sorted(eval_dict.keys()):
            log_string("eval {:>10s}: {:>4.4f} \t ({:>4d}/{:>4d})".format(key, eval_dict[key][0], eval_dict[key][1], eval_dict[key][2]))
            log_string("    \t (t:{:>4d}, r:{:>4d}, s:{:>4d} / {:>4d})".format(eval_dict[key][3], eval_dict[key][4], eval_dict[key][5], eval_dict[key][6]))

        print("------------------------------------------")
        log_string('eval class mean center:     %f'%(t))
        log_string('eval class mean rotation:   %f'%(r))
        log_string('eval class mean scale:      %f'%(s))
        print("------------------------------------------")
        log_string('eval class mean accuracy:   %f'%(class_mean_accuracy))
        print("------------------------------------------\n")

    return mean_loss, class_mean_accuracy


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    writer = SummaryWriter(LOG_DIR)
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))

        # Reset numpy seed.
        np.random.seed()
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = evaluate_one_epoch()

        # Write Loss per epoch
        if writer is not None:
            assert(epoch is not None)
            writer.add_scalars('Loss/Epoch', {'loss/train': train_loss, 'loss/val':val_loss}, epoch)
            if train_acc is not 0:
                writer.add_scalars('Acc/Epoch', {'acc/train': train_acc, 'acc/val':val_acc}, epoch)

        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
