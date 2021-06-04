import os, time
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder import PointNetAE
from Data.dataset import Dataset

from chamfer_distance.chamfer_distance import ChamferDistance 

import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from pdb import set_trace
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# Set CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

BASE_DIR = os.path.dirname(__file__)
DUMP_DIR = os.path.join(BASE_DIR, 'dump')

# Hyperparam
BATCH_SIZE = 32
data = 'shapenetcorev2'
n_points = 2048
latent = 512
NUM_NEIGHBORS = 3
dump = True

# Dataset/loader
TRAIN_DATASET = Dataset(dataset_name=data, num_points=n_points, split='train')
TEST_DATASET = Dataset(dataset_name=data, num_points=n_points, split='test')
EVAL_DATASET = Dataset(dataset_name=data, num_points=n_points, split='val')

TRAIN_DATALOADER = torch.utils.data.DataLoader(TRAIN_DATASET, 
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=4)
TEST_DATALOADER = torch.utils.data.DataLoader(TEST_DATASET,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=4)

# Set Network
n_dims = 3
net = PointNetAE(latent, n_dims, n_points)
net.to(device)
model = "./outputs/model512_50.pth"
net.load_state_dict(torch.load(model))
net.eval()


def retrieval(q, test=False):
    # Set data
    all_embeddings  = np.load(DUMP_DIR + '/all_embeddings.npy')
    all_labels      = np.load(DUMP_DIR + '/all_labels.npy')
    all_filenames   = np.load(DUMP_DIR + '/all_filenames.npy')

    with open(DUMP_DIR + '/shapenet_kdtree.pickle', 'rb') as pickle_file:
        database_kdtree = pickle.load(pickle_file)

        if not test:
            # Result dict
            pred = {}

            # Target data
            points, label, _, filename = TRAIN_DATASET[q]
            points = points.cuda().unsqueeze(0)
                
            embedding = net(points, r=True).detach().cpu()

            # Search nearest neighbor in embedding space
            dist, idx = database_kdtree.query(embedding, k=1)

            pred['label'] = all_labels[idx]
            pred['filename'] = all_filenames[idx]

        else:
            n = len(TRAIN_DATASET)
            for i in range(n):
                # Target data
                points, label, _, filename = TRAIN_DATASET[i]
                points = points.cuda().unsqueeze(0)
                
                embedding = net(points, r=True).detach().cpu()

                # Search nearest neighbor in embedding space
                label = label.cpu().item()
                filename = filename
                dist, q_pred_idx = database_kdtree.query(embedding, k=1)
                dist, e_pred_idx = database_kdtree.query(np.array([all_embeddings[i]]), k=1)

                # Output
                query_filename  = all_filenames[q_pred_idx][0][0][0]
                emb_filename = all_filenames[e_pred_idx][0][0][0]

                if filename != emb_filename or filename != query_filename:
                    print(i)
                    print(filename)
                    print(emb_filename)
                    print(query_filename)
                    pickle_file.close()
                    return 0

            print('success!')
            pickle_file.close()
            return 0

        pickle_file.close()
        
    return pred


if __name__ == "__main__":
    # collect_embedding()
    # tsne_visualization()
    output = retrieval(0, test=True)
