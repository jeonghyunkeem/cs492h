import os, sys
import json
import numpy as np
from numpy.core.defchararray import decode
from numpy.lib.arraypad import pad
import quaternion
import torch
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(DATA_DIR, 'Dataset')
DUMP_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

# import scan2cad_utils
from s2c_config import Scan2CADDatasetConfig
import s2c_utils

sys.path.append(os.path.join(ROOT_DIR, 'models/retrieval/'))
from Data.dataset import Dataset


DC = Scan2CADDatasetConfig()
MAX_NUM_POINT = 40000
MAX_NUM_OBJ = 64

INS_NUM_POINT = 2048

# NOT_CARED_ID = np.array([0])
INF = 9999
NOT_CARED_ID = np.array([0, 3])    # wall, floor, NONE

# Thresholds
PADDING = 0.05
SCALE_THRASHOLD = 0.05
SEG_THRESHOLD = 5

SYM2CLASS = {"__SYM_NONE": 0, "__SYM_ROTATE_UP_2": 1, "__SYM_ROTATE_UP_4": 2, "__SYM_ROTATE_UP_INF": 3}

# functions ==============================================================================================
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

def compose_mat4(t, q, s, center=None):
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M 

def decompose_mat4(M):
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s
# ========================================================================================================

class Scan2CADCollect(Dataset):
    def __init__(self, split_set='train', num_points=40000, export=False, augment=False, collect=False):
        self.data_path = os.path.join(DATA_DIR, 'Scan2CAD/export0')
        self.out_path = os.path.join(DATA_DIR, 'Scan2CAD/')

        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))

        self.scan_names = []
        if split_set in ['all', 'train', 'val', 'test']:
            split_filenames = os.path.join(BASE_DIR, 'scannet_meta',
                'scan2cad_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_list = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_list)
            self.scan_list = [sname for sname in self.scan_list \
                if sname in all_scan_names]
            print('Dataset for {}: kept {} scans out of {}'.format(split_set, len(self.scan_list), num_scans))
            num_scans = len(self.scan_list)
        else:
            print('illegal split name')
            return

        filename_json = BASE_DIR + "/full_annotations.json"
        assert filename_json
        self.dataset = {}
        with open(filename_json, 'r') as f:
            data = json.load(f)
            d = {}
            i = -1
            for idx, r in enumerate(data):
                i_scan = r["id_scan"]
                if i_scan not in self.scan_list: 
                    continue
                self.scan_names.append(i_scan)
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
                    d[i]['models'][j]['center'] = r["aligned_models"][j]['center']
                    d[i]['models'][j]['bbox'] = r["aligned_models"][j]['bbox']
                    d[i]['models'][j]['sym'] = SYM2CLASS[r["aligned_models"][j]['sym']]
                    d[i]['models'][j]['fname'] = r["aligned_models"][j]['id_cad']
                    cat_id = r["aligned_models"][j]['catid_cad']
                    d[i]['models'][j]['cat_id'] = cat_id
                    d[i]['models'][j]['sem_cls'] = DC.ShapenetIDtoClass(cat_id)                    

        self.dataset = d
        self.num_points = num_points
        self.augment = augment

        if collect:
            self.shapenetData = Dataset(dataset_name='shapenetcorev2', num_points=2048, split='all', class_choice=True, collect=True)

    def __len__(self):
        return len(self.dataset)
    
    def collect(self, N):
        """ Return dictionary of {verts(x,y,z): cad filename} 
            
            Note:
                NK = a total number of instances in dataset 
                V = a number of vertices
            
            args:
                N: int
                    a size of dataset
            
            return:
                dict: (NK, 1, V, 3)
                    a dictionary for verts-cad_file pairs
        """
        label = {}
        label['size'] = 0
        error_scan = {}
        for index in range(N):
            data = self.dataset[index]
            id_scan = data['id_scan']

            if index % 100 == 0:
                print('{:4d}: {}'.format(index, id_scan))

            K = data['n_total']
            assert(K <= MAX_NUM_OBJ)
            
            # Point Cloud
            mesh_vertices   = np.load(os.path.join(self.data_path, id_scan) + '_vert.npy')      # (N, 3)
            instance_labels = np.load(os.path.join(self.data_path, id_scan) + '_ins_label.npy') # (N, obj_id)
            semantic_labels = np.load(os.path.join(self.data_path, id_scan) + '_sem_label.npy') # (N, sem_cls)

            point_cloud = mesh_vertices[:,0:3]

            # Target CAD Objects (K, cls, 9 DoF)
            target_obj_classes      = np.zeros((MAX_NUM_OBJ), dtype=np.int64)
            target_obj_symmetry     = np.zeros((MAX_NUM_OBJ), dtype=np.int64)
            target_obj_alignments   = np.zeros((MAX_NUM_OBJ, 10), dtype=np.float32)    # trs, rot, scl
            target_obj_6d_rotation  = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
            target_obj_bbox         = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
            target_obj_verts        = np.zeros((MAX_NUM_OBJ, MAX_NUM_POINT, 3), dtype=np.float32)
            target_obj_verts_seg    = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3), dtype=np.float32)
            target_obj_verts_c      = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3), dtype=np.float32)
            target_obj_cad_points   = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3), dtype=np.float32)
            target_obj_paths = {}

            for model in range(K):
                # semantics
                target_obj_classes[model] = data['models'][model]['sem_cls']
                target_obj_symmetry[model] = np.array(data['models'][model]['sym'])

                # Transform
                obj_center = np.array(data['models'][model]['center'])
                obj_translation = np.array(data['models'][model]['trs']['translation'])
                obj_rotation = np.array(data['models'][model]['trs']['rotation'])
                obj_scale = np.array(data['models'][model]['trs']['scale'])
                if obj_scale[2] < SCALE_THRASHOLD:
                    obj_scale[2] = SCALE_THRASHOLD
                    print('     Scale violation => {} | {}'.format(id_scan, target_obj_classes[model]))
                Mobj = compose_mat4(obj_translation, obj_rotation, obj_scale, obj_center)
                t, q, s = decompose_mat4(Mobj)
                q = quaternion.as_float_array(q)
                target_obj_alignments[model, 0:3] = np.array(t)
                target_obj_alignments[model, 3:7] = np.array(q)
                target_obj_alignments[model, 7:10] = np.array(s)
                target_obj_6d_rotation[model, :] = from_q_to_6d(target_obj_alignments[model,3:7])

                # Boudning Box
                target_obj_bbox[model] = data['models'][model]['bbox']
                
                # Instance vertices
                # (1) Region Crop
                obj_corners = s2c_utils.get_3d_box_rotated(target_obj_bbox[model], Mobj, padding=PADDING)
                ex_points, obj_vert_ind = s2c_utils.extract_pc_in_box3d(point_cloud, obj_corners, [id_scan, target_obj_classes[model]])
                nx = ex_points.shape[0]
                target_obj_verts[model, :nx] = ex_points
                # (2) Instance Segments Crop
                seg_points = s2c_utils.filter_dominant_cls(ex_points, semantic_labels[obj_vert_ind], NOT_CARED_ID)
                seg_nx = seg_points.shape[0]
                if seg_nx < SEG_THRESHOLD:
                    scene_path = BASE_DIR + '/error/{}'.format(id_scan)
                    if not os.path.exists(scene_path): 
                        os.mkdir(scene_path)   
                    cls_path = scene_path + '/{}'.format(target_obj_classes[model])
                    if not os.path.exists(cls_path): 
                        os.mkdir(cls_path)  
                    s2c_utils.write_ply(points=ex_points, filename=os.path.join(cls_path, '{}_reg_points.ply'.format(model)))
                    s2c_utils.write_ply(points=seg_points, filename=os.path.join(cls_path, '{}_seg_points.ply'.format(model)))
                    continue
                if seg_nx > INS_NUM_POINT:
                    _, point_choices = s2c_utils.random_sampling(seg_points, INS_NUM_POINT, return_choices=True)   
                    seg_points = seg_points[point_choices]
                elif seg_nx < INS_NUM_POINT:
                    gap = INS_NUM_POINT - seg_nx
                    _, point_choices = s2c_utils.random_sampling(seg_points, gap, return_choices=True)
                    seg_points = np.concatenate((seg_points, seg_points[point_choices]), axis=0) 
                target_obj_verts_seg[model] = seg_points
                # (3) Canonical Instance Segments Crop
                hcoord = np.ones((INS_NUM_POINT, 1), dtype=np.float32)
                verts_seg_homogeneous = np.concatenate((seg_points, hcoord), axis=1)
                verts_seg_canonical   = np.dot(verts_seg_homogeneous, np.linalg.inv(Mobj).transpose())
                target_obj_verts_c[model] = verts_seg_canonical[:, :3]

                # CAD counterparts
                cat_id = data['models'][model]['cat_id']
                cad_name = data['models'][model]['fname']
                target_obj_paths[model] = cat_id + '/' + cad_name
                fname = target_obj_paths[model] + '.npy'
                target_obj_cad_points[model] = self.shapenetData.file_search(fname)
                # if model == 3:
                #     s2c_utils.write_ply(points=target_obj_verts_c[model], filename=os.path.join(BASE_DIR, 'canonical_check.ply'))

            target_bbox         = target_obj_bbox.copy()
            target_verts        = target_obj_verts.copy()
            target_verts_seg    = target_obj_verts_seg.copy()
            target_verts_canonical = target_obj_verts_c.copy()
            target_fpath        = target_obj_paths.copy()
            target_cad_points   = target_obj_cad_points.copy()
            target_cls          = target_obj_classes.copy()
            target_sym          = target_obj_symmetry.copy()

            # ----- GLOBAL LABEL -----
            K_before = label['size']
            error = 0
            error_code = {}
            K_ = 0
            for ins_id in range(K):
                cad_id = K_before + ins_id
                label[cad_id] = {}
                # vertices
                if np.sum(target_verts_seg[ins_id]) == 0:
                    error += 1
                    error_code[error] = DC.ClassToName[target_cls[ins_id]] 
                    continue
                label[cad_id]['v'] = target_verts[ins_id].astype(np.float32)
                label[cad_id]['vs'] = target_verts_seg[ins_id].astype(np.float32)
                label[cad_id]['vc'] = target_verts_canonical[ins_id].astype(np.float32)
                label[cad_id]['cad_v'] = target_cad_points[ins_id].astype(np.float32)
                # Alignments
                label[cad_id]['alignments'] = target_obj_alignments[ins_id].astype(np.float32)
                # semantic labels
                label[cad_id]['sem_cls'] = target_cls[ins_id].astype(np.int64)
                label[cad_id]['symmetry'] = target_sym[ins_id].astype(np.int64) 
                # file path
                label[cad_id]['fpath'] = target_fpath[ins_id] 
                K_ += 1
            label['size'] = K_before + K_

            if error > 0:
                error_scan[id_scan] = error_code
            
            # ----- SCENE LABEL -----
            scene_label = {}
            # Metadata
            scene_label['id_scan'] = id_scan
            scene_label['error'] = 1 if error > 0 else 0
            scene_label['n_total'] = np.array(K).astype(np.int64)
            # Points
            scene_label['point_cloud'] = point_cloud.astype(np.float32) # (N, 3)
            # Object Scans w/ CAD
            scene_label['v'] = target_verts[:K].astype(np.float32)             # (K, N, 3)
            scene_label['vs'] = target_verts_seg[:K].astype(np.float32)        # (K, 2048, 3)
            scene_label['vc'] = target_verts_canonical[:K].astype(np.float32)  # (K, 2048, 3)
            scene_label['cad_id'] = target_obj_paths    # {K: str}
            scene_label['cad_v'] = target_cad_points[:K].astype(np.float32) # (K, 2048, 3)
            # Object-CAD parameters
            scene_label['bbox'] = target_bbox[:K].astype(np.float32) # (K, 3)
            scene_label['alignment'] = target_obj_alignments[:K].astype(np.float32) # (K, 10)
            scene_label['sym'] = target_sym[:K].astype(np.float32)  # (K, )
            scene_label['sem_cls'] = target_cls[:K].astype(np.float32)  # (K, )

            scene_output = os.path.join(self.out_path, 'scene')
            np.save(os.path.join(scene_output, id_scan)+'.npy', scene_label)

        print('-'*30)
        print('total: {:4d} | error: {:4d}'.format(N, len(error_scan)))
        for i, (key, item) in enumerate(error_scan.items()):
            print('{:2d} {}: {}'.format(i, key, item))
        print('-'*30)

        np.save(self.out_path+'/scan-to-cad.npy', label)
        print('done | {}'.format(label['size']))

if __name__ == "__main__":
    Dataset = Scan2CADCollect(split_set='all', collect=True)
    N = len(Dataset)    
    Dataset.collect(N)


""" Discarded """
# ====================================================================================
# print(id_scan)
# print(np.unique(semantic_labels))
# print(np.unique(target_cls[:K]))

# # Lift Center (from Scannet to Scan2CAD)
# ddict = {}
# point_to_cad = {}
# for i_instance in np.unique(instance_labels):         
#     # find all points belong to that instance
#     ind = np.where(instance_labels == i_instance)[0]    
#     sem_cls = semantic_labels[ind[0]] # (0 ~ 21)

#     if sem_cls in NOT_CARED_ID:
#         continue

#     x = point_cloud[ind,:3]
#     center = 0.5*(x.min(0) + x.max(0))
#     # Find closest center of CAD pool
#     dist, cad_i = nn_search(center, target_translation[:K, 0:3])
#     # Update center to CAD center
#     # New cad model
#     if cad_i not in ddict:
#         ddict[cad_i] = dist
#         point_to_cad[cad_i] = ind
#     else:
#         # Update minimum distance
#         if dist < ddict[cad_i]:
#             ddict[cad_i] = dist
#             point_to_cad[cad_i] = ind

# # Instance Points
# ins_verts = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3))
# ins_verts_lifted = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3))
# ins_verts_h = np.ones((MAX_NUM_OBJ, INS_NUM_POINT, 1))
# ins_verts_canonical = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3))
# target_cad_points = np.zeros((MAX_NUM_OBJ, INS_NUM_POINT, 3))
# for cad_id in point_to_cad:
#     # Points within CAD instance
#     cad_ind = point_to_cad[cad_id]
#     cad_x = point_cloud[cad_ind, :3]
#     nx = cad_x.shape[0]
#     if nx > INS_NUM_POINT:
#         _, point_choices = s2c_utils.random_sampling(cad_x, INS_NUM_POINT, return_choices=True)   
#         cad_x = cad_x[point_choices]
#     elif nx < INS_NUM_POINT:
#         gap = INS_NUM_POINT - nx
#         _, point_choices = s2c_utils.random_sampling(cad_x, gap, return_choices=True)
#         cad_x = np.concatenate((cad_x, cad_x[point_choices]), axis=0) 

#     assert(cad_x.shape[0] == INS_NUM_POINT)
#     ins_verts[cad_id] = cad_x
#     cad_center = target_translation[cad_id]
#     cad_rotation = target_rotation[cad_id]
#     cad_scale = target_scale[cad_id]
#     cad_M = compose_mat4(cad_center, cad_rotation, cad_scale)
    
#     ins_verts_homogeneous = np.concatenate((ins_verts[cad_id], ins_verts_h[cad_id]), axis=1)
#     ins_verts_homogeneous_rot = np.dot(ins_verts_homogeneous, np.linalg.inv(cad_M).transpose())
#     ins_verts_canonical[cad_id] = ins_verts_homogeneous_rot[:, :3]

#     # Collect counterpart ShapeNetCore points
#     fname = target_obj_paths[cad_id] + '.npy'
#     target_cad_points[cad_id] = self.shapenetData.file_search(fname)

#     if index == 0 and cad_id == 2:
#         s2c_utils.write_ply(points=target_cad_points[cad_id], filename=os.path.join(BASE_DIR, 'cad_check.ply'))
#         s2c_utils.write_ply(points=ins_verts_canonical[cad_id], filename=os.path.join(BASE_DIR, 'canonical_check.ply'))
#         print(target_obj_paths[cad_id])

    # check_path = BASE_DIR + '/check/{}'.format(id_scan)
    # if not os.path.exists(check_path): 
    #     os.mkdir(check_path)   
    # s2c_utils.write_ply(points=seg_points, filename=os.path.join(check_path, '{}_verts.ply'.format(model)))
    # if model == 3:
    #     s2c_utils.write_ply(points=target_obj_cad_points[model], filename=os.path.join(check_path, 'cad_check.ply'))
    #     s2c_utils.write_ply(points=target_obj_verts_c[model], filename=os.path.join(check_path, 'canonical_check.ply'))
    #     print(target_obj_paths[cad_id])
    #
    # # only lift
    # cad_center = target_center[cad_id, 0:3]
    # ins_verts_lifted[cad_id] = ins_verts[cad_id] - cad_center
    # if target_cls[cad_id] == 3:
    #     i += 1
    #     if i > 1:
    #         # # Shapenet test
    #         # for item in range(len(self.shapenetData)):
    #         #     ps, _, _, fn = self.shapenetData[item]
    #         #     if fn == target_obj_paths[cad_id] + '.npy':
    #         #         ps = ps.cpu().numpy()
    #         #         # s2c_utils.write_ply(points=ps, filename=os.path.join(BASE_DIR, 'cad_org.ply'))
    #         #         pn = ps.shape[0]
    #         #         h = np.ones((pn, 1), dtype=np.float32)
    #         #         ps_h = np.concatenate((ps, h), axis=1)
    #         #         ps_rot = np.dot(ps_h, cad_M.transpose())
    #         #         ps = ps_rot[:, :3].copy()

    #         #         s2c_utils.write_ply(points=ps, filename=os.path.join(BASE_DIR, 'cad.ply'))
    #         #         print(target_obj_paths[cad_id])

    #         ins_verts_homogeneous = np.concatenate((ins_verts[cad_id], ins_verts_h[cad_id]), axis=1)
    #         ins_verts_homogeneous_rot = np.dot(ins_verts_homogeneous, np.linalg.inv(cad_M).transpose())
    #         ins_verts[cad_id] = ins_verts_homogeneous_rot[:, :3]
    #         s2c_utils.write_ply(points=ins_verts[cad_id, :nx], filename=os.path.join(BASE_DIR, 'vert_center.ply'))
    #         print(target_obj_paths[cad_id])
    #         return 0
    # ====================================================================================