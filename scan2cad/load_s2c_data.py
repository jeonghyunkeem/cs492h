import os, sys
import json
import numpy as np
import quaternion
import torch
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# import scan2cad_utils
from s2c_config import Scan2CADDatasetConfig
import s2c_utils

DC = Scan2CADDatasetConfig()
MAX_NUM_POINT = 40000
MAX_NUM_OBJ = 128

NOT_CARED_ID = np.array([0])

def nn_search(p, ps):
    target = torch.from_numpy(ps.copy()) 
    p = torch.from_numpy(p.copy())
    p_diff = target - p
    p_dist = torch.sum(p_diff**2, dim=-1)
    dist, idx = torch.min(p_dist, dim=-1)
    return dist.item(), idx.item()

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 

class Decoder(Dataset):
    def __init__(self):
        filename_json = BASE_DIR + "/full_annotations.json"
        assert filename_json

        self.dataset = {}
        self.scan_names = []
        with open(filename_json, 'r') as f:
            data = json.load(f)
            n_data = 0
            d = {}
            for i, r in enumerate(data):
                n_data += 1
                d[i] = {}
                
                i_scan = r["id_scan"]
                d[i]['id_scan'] = i_scan
                d[i]['trs'] = r["trs"]

                self.scan_names.append(i_scan)

        self.dataset = d

    def __len__(self):
        return len(self.dataset)

    def dump_list(self):
        output_file = open(BASE_DIR + '/scannet_meta/scan2cad.txt', 'w')
        output_file.write('\n'.join(sorted(self.scan_names)))
        output_file.close()

    def get_alignment_matrix(self, index):
        data = self.dataset[index]
        id_scan = data['id_scan']
        m_scan = make_M_from_tqs(data['trs']['translation'], data['trs']['rotation'], data['trs']['scale'])

        return id_scan, m_scan 