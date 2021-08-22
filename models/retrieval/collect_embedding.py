import os, sys, time
# from scan2cad.s2c_eval import ShapenetNameToClass
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from chamfer_distance.chamfer_distance import ChamferDistance 
from autoencoder import PointNetAE
from Data.dataset import Dataset
from Data.cat_map import ID2NAME, NAME2CLASS

import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from pdb import set_trace
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

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
DATASET = Dataset(dataset_name=data, num_points=n_points, split='all', class_choice=True)
DATALOADER = torch.utils.data.DataLoader(DATASET, 
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
    n = len(DATASET)
    print(n)
    pbar = tqdm(total=n, leave=False)
    
    # Evaluate
    total_loss = 0.0
    net.eval()

    # Set embeddings
    check = True
    for i, data in enumerate(DATALOADER):
        # Parse data
        points, label, category, filename = data
        points = points.cuda()

        # if check: return
        with torch.no_grad():
            recon_ret, embedding = net(points)

            embedding   = np.array(embedding.cpu())
            label       = np.array(label.cpu())
            category    = np.array(category).reshape(-1, 1)
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
            points, label, _, filename = DATASET[q]
            points = points.cuda().unsqueeze(0)
                
            embedding = net(points, r=True).detach().cpu()

            # Search nearest neighbor in embedding space
            dist, idx = database_kdtree.query(embedding, k=1)

            pred['label'] = all_labels[idx]
            pred['filename'] = all_filenames[idx]

        else:
            n = len(DATASET)
            for i in range(n):
                if i % 10000 == 0:
                    print(i)
                # Target data
                points, label, category, filename = DATASET[i]
                points = points.cuda().unsqueeze(0)
                
                embedding = net(points, r=True).detach().cpu()
                gt_embedding = np.array([all_embeddings[i]])

                # Search nearest neighbor in embedding space
                # label = label.cpu().item()
                # filename = filename
                dist, q_pred_idx = database_kdtree.query(embedding, k=1)
                dist, e_pred_idx = database_kdtree.query(gt_embedding, k=1)

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

def tsne_visualization(ppl=30):
    all_embeddings  = np.load(DUMP_DIR + '/all_embeddings.npy')
    all_category    = np.load(DUMP_DIR + '/all_category.npy')
    all_category = all_category.reshape(-1).tolist()

    time_start = time.time()
    print('Start t-SNE...')
    tsne = TSNE(n_components=2, perplexity=ppl, random_state=0)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    # Plot
    plt.figure(figsize=(16,16))
    df = pd.DataFrame()
    df["x"] = embeddings_2d[:,0]
    df["y"] = embeddings_2d[:,1]

    # all_category = all_category.reshape(-1).tolist()
    color_map = cm.rainbow(np.linspace(0, 1, 35))
    colors = []
    for i, cat in enumerate(all_category):
        all_category[i] = ID2NAME[cat]
        color = color_map[NAME2CLASS[ID2NAME[cat]]]
        colors.append(color)
    labels = np.unique(np.array(all_category))
    # plt.scatter(x=df.x, y=df.y, c=colors, label=labels, s=0.5, alpha=0.5, data=df)
    sns.scatterplot(x=df.x, y=df.y, hue=all_category, 
                    palette=sns.color_palette('hls', 35), 
                    data=df,
                    legend='full', 
                    alpha=0.5)

    # plt.legend()
    file_name = 'tsne_512_ppl' + str(ppl) + '.png'
    plt.savefig(os.path.join(DUMP_DIR, file_name))

def establish(test=True):
    collect_embedding()
    output = retrieval(0, test=test)

if __name__ == "__main__":
    # establish(test=True)
    tsne_visualization(ppl=50)
