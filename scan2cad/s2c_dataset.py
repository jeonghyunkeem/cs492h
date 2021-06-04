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
from scan2cad.s2c_config import Scan2CADDatasetConfig
import s2c_utils

DC = Scan2CADDatasetConfig()
MAX_NUM_POINT = 40000
MAX_NUM_OBJ = 64

NOT_CARED_ID = np.array([0])

SYM2CLASS = {"__SYM_NONE": 0, "__SYM_ROTATE_UP_2": 1, "__SYM_ROTATE_UP_4": 2, "__SYM_ROTATE_UP_INF": 3}

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def from_q_to_6d(q):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    mat = quaternion.as_rotation_matrix(q)  # 3x3
    rep6d = mat[:, 0:2].transpose().reshape(-1, 6)   # 6
    return rep6d

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

class Scan2CADDataset(Dataset):
    def __init__(self, split_set='train', num_points=40000, augment=False):
        self.data_path = os.path.join(BASE_DIR, 'scannet_data')
        filename_json = BASE_DIR + "/full_annotations.json"
        assert filename_json

        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))

        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(BASE_DIR, 'scannet_meta',
                'scan2cad_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('Dataset for {}: kept {} scans out of {}'.format(split_set, len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return

        self.dataset = {}
        self.test: int = None
        with open(filename_json, 'r') as f:
            data = json.load(f)
            d = {}
            i = -1
            for idx, r in enumerate(data):
                i_scan = r["id_scan"]
                
                if i_scan not in self.scan_names:
                    continue
                
                i += 1
                d[i] = {}
                d[i]['id_scan'] = i_scan
                d[i]['trs'] = r["trs"]
                
                n_model = r["n_aligned_models"]
                d[i]['n_total'] = n_model
                d[i]['models'] = {}
                for j in range(n_model):
                    d[i]['models'][j] = {}
                    d[i]['models'][j]['trs'] = r["aligned_models"][j]['trs']
                    d[i]['models'][j]['sym'] = SYM2CLASS[r["aligned_models"][j]['sym']]

                    cat_id = r["aligned_models"][j]['catid_cad']
                    if cat_id in DC.ShapenetIDToName:
                        d[i]['models'][j]['sem_cls'] = DC.ShapenetIDToName[cat_id]
                    else:
                        d[i]['models'][j]['sem_cls'] = 'other'

                    d[i]['models'][j]['id'] = r["aligned_models"][j]['id_cad']

        self.dataset = d
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):  
        data = self.dataset[index]
        id_scan = data['id_scan']
        K = data['n_total']
        assert(K <= MAX_NUM_OBJ)
        
        # Point Cloud
        mesh_vertices   = np.load(os.path.join(self.data_path, id_scan) + '_vert.npy')      # (N, 3)
        instance_labels = np.load(os.path.join(self.data_path, id_scan) + '_ins_label.npy') # (N, obj_id)
        semantic_labels = np.load(os.path.join(self.data_path, id_scan) + '_sem_label.npy') # (N, sem_cls)
        instance_bboxes = np.load(os.path.join(self.data_path, id_scan) + '_bbox.npy')      # (obj_id, 7)

        point_cloud = mesh_vertices[:,0:3] # do not use color for now
        point_cloud, choices = s2c_utils.random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        # Target CAD Objects (K, cls, 9 DoF)
        target_obj_classes = np.zeros((MAX_NUM_OBJ), dtype=np.int64)
        target_obj_symmetry = np.zeros((MAX_NUM_OBJ), dtype=np.int64)
        target_obj_alignments = np.zeros((MAX_NUM_OBJ, 10), dtype=np.float32)    # trs, rot, scl
        target_obj_6d_rotation = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        for model in range(K):
            # Class (1)
            cls_str = data['models'][model]['sem_cls']
            target_obj_classes[model] = int(DC.ShapenetNameToClass[cls_str])
            # Center (3)
            target_obj_alignments[model, 0:3] = np.array(data['models'][model]['trs']['translation'])
            # Rotation (4)
            target_obj_alignments[model, 3:7] = np.array(data['models'][model]['trs']['rotation'])
            # 6D representation
            target_obj_6d_rotation[model, :] = from_q_to_6d(target_obj_alignments[model,3:7])
            # Scale (3)
            target_obj_alignments[model, 7:10] = np.array(data['models'][model]['trs']['scale'])
            # Symmetry (1)
            target_obj_symmetry[model] = np.array(data['models'][model]['sym'])

        target_cls      = target_obj_classes.copy()
        target_center   = target_obj_alignments[:, 0:3].copy()
        target_rotation = target_obj_alignments[:, 3:7].copy()
        target_scale    = target_obj_alignments[:, 7:10].copy()
        target_sym      = target_obj_symmetry.copy()

        # # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            # Rotation along up-axis/Y-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = roty(rot_angle)
            rot_mat_T = np.transpose(rot_mat)

            # Rotate point cloud
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], rot_mat_T)

            # Rotate Center
            target_center[:,0:3] = np.dot(target_center[:,0:3], rot_mat_T)

            # Rotate Rotation
            target_R = np.zeros((K, 3, 3))
            rot_mat_q = roty(-rot_angle)
            rot_mat_q_T = np.transpose(rot_mat_q)

            for model in range(K):
                # --- Rotation-Quaternion ---
                # Convert quaternion to rotation matrix
                target_R[model, :, :] = np.eye(3)
                q0 = target_rotation[model, 0:4]
                target_q0 = np.quaternion(q0[0], q0[1], q0[2], q0[3])
                target_R[model, 0:3, 0:3] = quaternion.as_rotation_matrix(target_q0)
                
                # Update Rotation
                target_R[model, :, :] = np.dot(target_R[model, :, :], rot_mat_q_T)
                gt_q = quaternion.from_rotation_matrix(target_R[model, :, :])
                gt_q_array = quaternion.as_float_array(gt_q)
                target_rotation[model, :] = gt_q_array

                # Rotation-6D Representation
                target_obj_6d_rotation[model, :] = from_q_to_6d(target_rotation[model, :])

        # ------ VOTES ------
        # Generate Votes 
        point_votes             = np.zeros([self.num_points, 3])
        point_votes_mask        = np.zeros(self.num_points)

        ddict = {}
        point_to_cad = {}
        for i_instance in np.unique(instance_labels):         
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]    
            sem_cls = semantic_labels[ind[0]] # (0 ~ 21)
            # find the semantic label            
            if sem_cls in NOT_CARED_ID:
                continue
            else:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                # Find closest center of CAD
                dist, cad_i = nn_search(center, target_center[:K, 0:3])
                # i_cls = target_cls[cad_i]
                # if i_cls != (sem_cls-1): 
                #     continue
                # Dictionary for update center to cad center
                # New cad model
                if cad_i not in ddict:
                    ddict[cad_i] = dist
                    point_to_cad[cad_i] = ind
                else:
                    # Update minimum distance if so
                    if dist < ddict[cad_i]:
                        ddict[cad_i] = dist
                        point_to_cad[cad_i] = ind

        # Target Bounding Boxes
        target_bboxes_size = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))   
    
        # assert len(point_to_cad) == K
        for cad_id in point_to_cad:
            # Points within CAD instance
            cad_ind = point_to_cad[cad_id]
            cad_x = point_cloud[cad_ind, 0:3]
            cad_center = target_center[cad_id, 0:3]
            # Bounding box label
            target_bboxes_mask[cad_id] = 1.0
            bbox_length = abs(cad_x.max(0) - cad_center)*2
            target_bboxes_size[cad_id, :] = bbox_length
            # Update center to cad center
            point_votes[cad_ind, :] = cad_center - cad_x
            point_votes_mask[cad_ind] = 1.0

        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 

        # ----- LABEL -----
        label = {}
        label['point_clouds'] = point_cloud.astype(np.float32)
        label['sem_cls_label'] = target_cls.astype(np.int64)
        label['cad_sym_label'] = target_sym.astype(np.int64)
        label['box_mask_label'] = target_bboxes_mask.astype(np.float32)
        label['box_size_label'] = target_bboxes_size.astype(np.float32) 
        label['center_label'] = target_center.astype(np.float32)
        label['rotation_label'] = target_rotation.astype(np.float32)
        label['rot_6d_label'] = target_obj_6d_rotation.astype(np.float32)
        label['scale_label'] = target_scale.astype(np.float32)
        label['vote_label'] = point_votes.astype(np.float32)
        label['vote_label_mask'] = point_votes_mask.astype(np.int64)
        label['n_total'] = np.array(K).astype(np.int64)

        return label

def test_augmenatation(angle):
    return 0

if __name__ == "__main__":
    Dataset = Scan2CADDataset()
    N = len(Dataset)    
    for t in range(N):
        Dataset.__getitem__(t)

    # test = 3
    # for t in range(test):
    #     print("---- Test {} ----".format((t+1)*10))
    #     Dataset.__getitem__(t)