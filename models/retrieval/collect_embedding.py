import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from chamfer_distance.chamfer_distance import ChamferDistance 
from autoencoder import PointNetAE
from Data.dataset import Dataset

import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from pdb import set_trace
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DUMP_DIR = os.path.join(BASE_DIR, 'dump')
sys.path.append(os.path.join(BASE_DIR, 'chamfer_distance'))



# Hyperparam
BATCH_SIZE = 32
data = 'shapenetcorev2'
n_points = 2048
latent = 512
NUM_NEIGHBORS = 3
dump = True

# Set CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataset/loader
TRAIN_DATASET = Dataset(dataset_name=data, num_points=n_points, split='train', class_choice=True)
TEST_DATASET = Dataset(dataset_name=data, num_points=n_points, split='test', class_choice=True)
EVAL_DATASET = Dataset(dataset_name=data, num_points=n_points, split='val', class_choice=True)

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
model = os.path.join(BASE_DIR, 'outputs') + "/model512_50.pth"
net.load_state_dict(torch.load(model))
net.eval()

# # Set Loss function
# chamfer_distance = ChamferDistance()
# criterion = chamfer_distance

def collect_embedding():    
    n = len(TRAIN_DATASET)
    pbar = tqdm(total=n, leave=False)
    
    # Evaluate
    total_loss = 0.0
    net.eval()

    # Set embeddings
    check = True
    for i, data in enumerate(TRAIN_DATALOADER):
        # Parse data
        points, label, category, filename = data
        points = points.cuda()

        # if check: return
        with torch.no_grad():
            recon_ret, embedding = net(points)

            embedding   = np.array(embedding.cpu())
            label       = np.array(label.cpu())
            category    = np.array(category.cpu()).reshape(-1, 1)
            filename    = np.array(list(filename)).reshape(-1, 1)

            # if check: return
            if i == 0: 
                all_embeddings  = np.array(embedding)
                all_labels      = np.array(label)
                all_category    = np.array(category)
                all_filenames   = np.array(filename)
            else: 
                all_embeddings  = np.vstack((all_embeddings, embedding))
                all_labels      = np.vstack((all_labels, label))
                all_category    = np.vstack((all_category, category))
                all_filenames   = np.vstack((all_filenames, filename))

            # dist1, dist2 = criterion(points, recon_ret)
            # loss = (torch.mean(dist1)) + (torch.mean(dist2))
            
        batch_size = list(data[0].size())[0]
        # total_loss += loss * batch_size
        
        # pbar.set_description('Train Loss: {:f}'.format(loss))
        pbar.update(batch_size)
        
    pbar.close()

    np.save(DUMP_DIR + '/all_embeddings', all_embeddings)
    np.save(DUMP_DIR + '/all_labels', all_labels)
    np.save(DUMP_DIR + '/all_category', all_category)
    np.save(DUMP_DIR + '/all_filenames', all_filenames)

    ####For Retrieval
    if dump:
        database_kdtree = KDTree(all_embeddings)
        pickle_out = open(os.path.join(DUMP_DIR, "shapenet_kdtree.pickle"),"wb")
        pickle.dump(database_kdtree, pickle_out)
        pickle_out.close()


def tsne_visualization():
    all_embeddings  = np.load(DUMP_DIR + '/all_embeddings.npy')
    all_labels      = np.load(DUMP_DIR + '/all_labels.npy')

    time_start = time.time()
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, label=all_labels, s=0.2, alpha=0.5)

    plt.legend()
    plt.savefig(os.path.join(DUMP_DIR, 'tsne.png'))


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
                points, label, category, filename = TRAIN_DATASET[i]
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
    collect_embedding()
    # tsne_visualization()
    output = retrieval(0, test=True)
