#!/usr/bin/env python
# coding: utf-8

# CS492H 2021 Prof. M Sung
# PA2 AutoEncoder
# Jeonghyun Kim

import os, sys, argparse
import h5py
# import easydict
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder import PointNetAE
from Data.dataset import Dataset

# @Ref: https://github.com/chrdiller/pyTorchChamferDistance
from chamfer_distance.chamfer_distance import ChamferDistance 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_points', type=int, default=2048, help="number of points to sample")
parser.add_argument('--latent', type=int, default=512, help="dimension of latent space")
parser.add_argument('--train', type=bool, default=True, help='set true if train')
parser.add_argument('--batch_size', type=int, default=32, help="input batch size")
parser.add_argument('--epoch', type=int, default=50, help="number of epoch")
parser.add_argument('--n_workers', type=int, default=4, help="number of data loading workers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--b1', type=float, default=0.9, help="beta 1")
parser.add_argument('--b2', type=float, default=0.999, help="beta 2")
parser.add_argument('--step_size', type=int, default=20, help="step size")
parser.add_argument('--gamma', type=float, default=0.5, help="gamma")

# I/O
parser.add_argument('--data', type=str, 
                    default='shapenetcorev2', help="name of dataset")
parser.add_argument('--model', type=str, default='', help="model path")
parser.add_argument('--out_dir', type=str, default='outputs', help="output directory")

args = parser.parse_args()

TRAIN_DATASET = Dataset(dataset_name=args.data, num_points=args.n_points, split='train')
TEST_DATASET = Dataset(dataset_name=args.data, num_points=args.n_points, split='test')
EVAL_DATASET = Dataset(dataset_name=args.data, num_points=args.n_points, split='val')

TRAIN_DATALOADER = torch.utils.data.DataLoader(TRAIN_DATASET, 
                                                   batch_size=args.batch_size,
                                                   shuffle=args.train,
                                                   num_workers=int(args.n_workers))
TEST_DATALOADER = torch.utils.data.DataLoader(TEST_DATASET,
                                                  batch_size=args.batch_size,
                                                  shuffle=args.train,
                                                  num_workers=int(args.n_workers))    


# Set Network
n_dims = 3
net = PointNetAE(args.latent, n_dims, args.n_points)
net.to(device)

# Load model if there is any    
if args.model != '':
    net.load_state_dict(torch.load(args.model))

# Set Loss function
chamfer_distance = ChamferDistance()
criterion = chamfer_distance

# Set optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Set Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


def train_one_epoch(epoch=None, writer=None, epoch_str=None):    
    # Create a Progress bar.
    n = len(TRAIN_DATASET)
    pbar = tqdm(total=n, leave=False)
    
    # Train
    total_loss = 0.0
    net.train()
    for i, data in enumerate(TRAIN_DATALOADER):
        # Parse data
        points, label, category, filename = data
        points = points.cuda()  # (B, N, 3)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Reconstruction
        recon_ret, _ = net(points)
        
        # Compute Loss
        dist1, dist2 = criterion(points, recon_ret)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        
        # Update Parameters
        loss.backward()
        optimizer.step()

        # Write results
        if writer is not None:
            assert(epoch is not None)
            step = epoch * len(TRAIN_DATALOADER) + i
            writer.add_scalar('Loss/Train', loss, step)
            
        batch_size = list(data[0].size())[0]
        total_loss += loss * batch_size
        
        pbar.set_description('{} Train Loss: {:f}'.format(epoch_str, loss))
        pbar.update(batch_size)
    
    # Close progress bar
    pbar.close()
    
    mean_loss = total_loss / n
    
    return mean_loss


def dump_result(points, filename):
    """
        Args:
            points: (N, 3)
            filename: str
        
        Returns:
            Save file with filename.ply
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def eval_one_epoch(epoch_str=None, dump=False):    
    n = len(TEST_DATASET)
    pbar = tqdm(total=n, leave=False)
    
    # Evaluate
    total_loss = 0.0
    net.eval()
    for i, data in enumerate(TEST_DATALOADER):
        # Parse data
        points, label, category, filename = data
        points = points.cuda()
        
        with torch.no_grad():
            recon_ret, _ = net(points)
            
            dist1, dist2 = criterion(points, recon_ret)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            
        batch_size = list(data[0].size())[0]
        total_loss += loss * batch_size
        
        pbar.set_description('{} Train Loss: {:f}'.format(epoch_str, loss))
        pbar.update(batch_size)
        
        if dump:
            dump_result(points[0, :, :], './result/{}_{}_points.ply'.format(i, args.latent))
            dump_result(recon_ret[0, :, :], './result/{}_{}_recon.ply'.format(i, args.latent))
        
    pbar.close()
    
    mean_loss = total_loss / n
    
    return mean_loss


def train(n_epoch):    
    writer = SummaryWriter(args.out_dir)
    
    for epoch in range(n_epoch):
        epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(str(epoch).zfill(len(str(args.epoch))), args.epoch)
        
        # Compute Loss
        train_loss = train_one_epoch(epoch=epoch,
                                     writer=writer,
                                     epoch_str=epoch_str)
        test_loss = eval_one_epoch(epoch_str=epoch_str)
        
        # Update Scheduler
        scheduler.step()
        
        if writer is not None:
            # Write test results.
            assert(epoch is not None)
            step = (epoch + 1) * len(TRAIN_DATALOADER)
            writer.add_scalar('Loss/Test', test_loss, step)
        
        # Log statistics
        log = epoch_str + ' '
        log += 'Train Loss: {:f}, '.format(train_loss)
        log += 'Test Loss: {:f}, '.format(test_loss)
        print(log)
        
        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_file = os.path.join(args.out_dir, 'model{}_{:d}.pth'.format(args.latent, epoch + 1))
            torch.save(net.state_dict(), model_file)
            print("Saved '{}'.".format(model_file))
        
    writer.close()


def eval():
    # Compute Loss
    test_loss = eval_one_epoch(dump=True)
    
    # Log statistics
    log = ''
    log += 'Avg Chamfer Distance: {:f} for {}'.format(test_loss, args.model)
    print(log)


if __name__ == "__main__":
    print(args)
    
    # Create the output directory.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    if args.train:
        train(args.epoch)
    else:
        eval()