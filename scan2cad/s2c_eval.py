import torch
import numpy as np
import quaternion
import os
import s2c_utils
import pickle

ShapenetNameToClass = {'chair': 0, 'table': 1, 'cabinet': 2, 'trash bin': 3, 'bookshelf': 4,'display': 5,'sofa': 6, 'bathtub': 7, 'other': 8}
ShapenetClassToName = {ShapenetNameToClass[k]: k for k in ShapenetNameToClass}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RET_DIR = os.path.join(ROOT_DIR, 'models/retrieval/dump')
from models.retrieval.autoencoder import PointNetAE

def from_6d_to_mat(v):
    v1 = v[:,:,:3].unsqueeze(-1)  # (B, K, 3, 1)
    v2 = v[:,:,3:].unsqueeze(-1)  # (B, K, 3, 1)
    v3 = torch.cross(v1, v2, dim=2)
    M = torch.stack([v1, v2, v3], dim=3).squeeze(-1)
    return M

def from_mat_to_q(M):
    R = M[:, :, 0:3, 0:3].detach().cpu().numpy().copy()
    q = quaternion.from_rotation_matrix(R[:, :, 0:3, 0:3])

    return q

def from_6d_to_q(v):
    M = from_6d_to_mat(v)
    q = from_mat_to_q(M)
    
    return q

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

# helper function to calculate difference between two quaternions 
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation

def get_top_8_category(cat_id):
    if cat_id > 7:
        cat = 8
    else:
        cat = cat_id

    return cat

class Evaluation:
    def __init__(self, nms_iou=0.25):
        self.class_total = {}
        self.pred_total = {}
        self.acc_per_scan = {}
        self.acc_proposal_per_class = {}
        self.acc_translation_per_class = {}
        self.acc_rotation_per_class = {}
        self.acc_scale_per_class = {}

        for i in ShapenetClassToName:
            self.class_total[i] = 0
            self.pred_total[i]= 0
            self.acc_proposal_per_class[i] = 0
            self.acc_translation_per_class[i] = 0
            self.acc_rotation_per_class[i] = 0
            self.acc_scale_per_class[i] = 0

        self.validate_idx_per_scene = {}
        self.nms_iou = nms_iou
        self.extra_dict = {}

        # CAD Retrieval
        self.sem_clses = np.load(RET_DIR + '/all_category.npy')
        print(np.unique(self.sem_clses))
        self.filenames = np.load(RET_DIR + '/all_filenames.npy')
        self.CADnet = PointNetAE(latent=512, in_dims=3, n_points=2048)
        self.CADnet.load_state_dict(torch.load(RET_DIR + '/model512_50.pth'))
        self.CADnet.cuda()
        self.CADnet.eval()


    def NMS(self, B, K, center, size, obj_prob, sem_cls):
        pred_crnrs_3d_upright_cam = np.zeros((B, K, 8, 3))
        for b in range(B):
            for k in range(K):
                crnrs_3d_upright_cam = s2c_utils.get_3d_box(size[b,k,:3], 0, center[b,k,:3])
                pred_crnrs_3d_upright_cam[b,k] = crnrs_3d_upright_cam
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((B, K))
        nonempty_box_mask = np.ones((B, K))
        for i in range(B):
            boxes_3d_with_prob = np.zeros((K,8))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_crnrs_3d_upright_cam[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_crnrs_3d_upright_cam[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_crnrs_3d_upright_cam[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_crnrs_3d_upright_cam[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_crnrs_3d_upright_cam[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_crnrs_3d_upright_cam[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
                boxes_3d_with_prob[j,7] = sem_cls[i,j] # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = s2c_utils.nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:], self.nms_iou)
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1

        return pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    def step(self, end_points, batch_iter, pcd=None):
        gt_center = end_points['center_label']
        pred_center = end_points['center']
        B = gt_center.shape[0]      # Batch size
        K = pred_center.shape[1]    # Num proposals
        K2 = gt_center.shape[1]   # Max obj num
        
        # Ground-truth
        gt_center       = end_points['center_label'].detach().cpu().numpy()    # (B, K2, 3)
        gt_quaternion   = end_points['rotation_label'].detach().cpu().numpy()  # (B, K2, 4)
        gt_scale        = end_points['scale_label'].detach().cpu().numpy()     # (B, K2, 3)
        gt_class        = end_points['sem_cls_label'].reshape(B, K2, 1)   # (B, K2, 1)
        gt_cad_total    = end_points['n_total'].reshape(B, 1)         # (B, 1)
        gt_sym_label    = end_points['cad_sym_label'].reshape(B, K2, 1)

        # prediction
        pred_class      = torch.argmax(end_points['sem_cls_scores'], -1).reshape(B, K, 1)
        pred_center     = end_points['center'].detach().cpu().numpy() 
        pred_size       = end_points['box_size'].detach().cpu().numpy()

        pred_rot_6d     = end_points['rot_6d_scores'] # (B, K, 6)
        pred_quaternion = from_6d_to_q(pred_rot_6d)
        # pred_quaternion = end_points['rotation_scores'].detach().cpu().numpy() # (B, K, 4)

        pred_scale      = end_points['scale_scores'].detach().cpu().numpy()    
        pred_obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
        
        pred_obj = softmax(pred_obj_logits)[:,:,1] # (B,K)
        pred_mask = self.NMS(B, K, pred_center, pred_scale, pred_obj, pred_class)

        # Threshold
        threshold_translation = 0.2 # <-- in meter
        threshold_rotation = 20 # <-- in deg
        threshold_scale = 20 # <-- in %

        class_total = {}
        pred_total = {}
        acc_proposal_per_class = {}
        acc_translation_per_class = {}
        acc_rotation_per_class = {}
        acc_scale_per_class = {}

        for i in ShapenetClassToName:
            class_total[i] = 0
            pred_total[i] = 0
            acc_proposal_per_class[i] = 0
            acc_translation_per_class[i] = 0
            acc_rotation_per_class[i] = 0
            acc_scale_per_class[i] = 0

        # Change category
        for b in range(B):
            
            for k in range(K):
                pred_class[b,k,:] = get_top_8_category(pred_class[b,k,:])
            for k2 in range(K2):
                gt_class[b,k2,:] = get_top_8_category(gt_class[b,k2,:])

        acc_per_scan = {}
        for b in range(B):
            acc_per_scan[batch_iter + b] = {}
            acc_per_scan[batch_iter + b]['n_total'] = gt_cad_total[b]
            acc_per_scan[batch_iter + b]["n_good"] = 0
            acc_per_scan[batch_iter + b]["n_files"] = []

            self.validate_idx_per_scene[b] = []
            K2 = gt_cad_total[b] # GT_cad_in_one_scene
            for k_gt in range(K2):
                # ------ Update total ------
                gt_sem_cls = gt_class[b,k_gt,:].item()

                if gt_sem_cls not in class_total:
                    class_total[gt_sem_cls] = 1
                else:
                    class_total[gt_sem_cls] += 1

        # CAD Retrieval
        if pcd is not None:
            batch_pc = pcd.cpu().numpy()[:,:,0:3]   # (B, N, 3)

        for b in range(B):  # loop in scenes
            K2 = gt_cad_total[b] # GT_cad_in_one_scene
            pred_gt = []
            with open(RET_DIR + '/shapenet_kdtree.pickle', 'rb') as pickle_file:
                database_kdtree = pickle.load(pickle_file)
                for k in np.where(pred_mask[b, :] == 1)[0]:  # loop in proposals           
                    # Class prediction
                    if pcd is not None:
                        box3d = s2c_utils.get_3d_box(pred_size[b,k,:3], 0, pred_center[b,k,:3])
                        box3d = s2c_utils.flip_axis_to_depth(box3d)
                        pc_in_box, inds = s2c_utils.extract_pc_in_box3d(batch_pc[b,:,:3], box3d)
                        if len(pc_in_box) < 5:
                            continue
                        cad_inds = np.where(inds == True)
                        cad_pc = pcd[b, cad_inds, :3]
                        embedding = self.CADnet(cad_pc, r=True)
                        embedding = embedding.detach().cpu()
                        dist, pred_idx = database_kdtree.query(embedding, k=5)
                        # Output
                        pred_sem_clses = self.sem_clses[pred_idx].squeeze(0)
                        cad_files      = self.filenames[pred_idx].squeeze(0)
                    else:
                        pred_sem_cls = pred_class[b, k, :][0]
                    for k_gt in range(K2):
                        # Pass predicted ground-truth
                        if k_gt in pred_gt: continue

                        # ------ Compare Prediction with GT ------
                        gt_sem_cls = gt_class[b,k_gt,:].item()
                        pred_sem_cls = -1
                        # Only compare with same class
                        for i in range(5):
                            if pred_sem_clses[i] > 8: 
                                pred_sem_clses[i] = 8 
                            if pred_sem_clses[i] == gt_sem_cls:
                                pred_sem_cls = pred_sem_clses[i, 0:1]
                                cad_file = cad_files[i, 0:1]

                        is_same_class = pred_sem_cls == gt_sem_cls
                        if is_same_class:
                            pred_total[gt_sem_cls] += 1
                            # Predicted Transformation
                            c = pred_center[b,k,:]
                            # q0 = pred_quaternion[b,k]
                            # q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
                            q = pred_quaternion[b,k]
                            #                                                                                                                                                           q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
                            s = pred_scale[b,k,:]

                            # Ground-truth Transformation
                            c_gt = gt_center[b,k_gt,:]
                            q_gt0 = gt_quaternion[b,k_gt,:]
                            q_gt = np.quaternion(q_gt0[0], q_gt0[1], q_gt0[2], q_gt0[3])
                            s_gt = gt_scale[b,k_gt,:]

                            # ---- Compute Error ----
                            # CENTER
                            error_translation = np.linalg.norm(c-c_gt, ord=2)
                            if error_translation <= threshold_translation: 
                                acc_translation_per_class[gt_sem_cls] += 1

                            # SCALE
                            error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)
                            if error_scale <= threshold_scale: 
                                acc_scale_per_class[gt_sem_cls] += 1

                            # ROTATION
                            sym = gt_sym_label[b, k_gt].item()
                            if sym == 1:
                                m = 2
                                tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                                error_rotation = np.min(tmp)
                            elif sym == 2:
                                m = 4
                                tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                                error_rotation = np.min(tmp)
                            elif sym == 3:
                                m = 36
                                tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                                error_rotation = np.min(tmp)
                            else:
                                error_rotation = calc_rotation_diff(q, q_gt)

                            if error_rotation <= threshold_rotation:
                                acc_rotation_per_class[gt_sem_cls] += 1
                            
                            # CHECK ANSWER
                            is_valid_transformation = error_rotation <= threshold_rotation and error_translation <= threshold_translation and error_scale <= threshold_scale

                            if is_valid_transformation:
                                acc_per_scan[batch_iter + b]["n_good"] += 1
                                acc_per_scan[batch_iter + b]["n_files"].append(cad_file)
                                if gt_sem_cls not in acc_proposal_per_class:
                                    acc_proposal_per_class[gt_sem_cls] = 1
                                else:
                                    acc_proposal_per_class[gt_sem_cls] += 1
                                
                                self.validate_idx_per_scene[b].append(k)
                                pred_gt.append(k_gt)
                                break
                
        # print(acc_per_scan)
        # Update
        for b in range(B):
            b_id_scan = batch_iter + b
            self.acc_per_scan[b_id_scan] = {}
            self.acc_per_scan[b_id_scan]["n_total"] = acc_per_scan[b_id_scan]["n_total"].item()
            self.acc_per_scan[b_id_scan]["n_good"] = acc_per_scan[b_id_scan]["n_good"]
            self.acc_per_scan[b_id_scan]["n_files"] = acc_per_scan[b_id_scan]["n_files"]

        for sem_cls, n_total in class_total.items():
            self.class_total[sem_cls]               += n_total
            self.pred_total[sem_cls]                += pred_total[sem_cls]
            self.acc_proposal_per_class[sem_cls]    += acc_proposal_per_class[sem_cls]
            self.acc_translation_per_class[sem_cls] += acc_translation_per_class[sem_cls]
            self.acc_rotation_per_class[sem_cls]    += acc_rotation_per_class[sem_cls]
            self.acc_scale_per_class[sem_cls]       += acc_scale_per_class[sem_cls]

        
    def summary(self):
        eval_dict = {}
        accuracy_per_class = {}
        good_t_per_class = {}
        good_r_per_class = {}
        good_s_per_class = {}

        # Per scan
        total_accuracy = {"n_total": 0, "n_good": 0}
        for id_scan in self.acc_per_scan:
            total_accuracy["n_total"] += self.acc_per_scan[id_scan]["n_total"]
            total_accuracy["n_good"] += self.acc_per_scan[id_scan]["n_good"]
        
        instance_mean_accuracy = float(total_accuracy["n_good"])/total_accuracy["n_total"]

        # Per class
        for sem_cls, n_total in self.class_total.items():
            cat_name = ShapenetClassToName[sem_cls]
            prediction = self.acc_proposal_per_class[sem_cls]
            accuracy_per_class[sem_cls] = float(prediction / (n_total + 1e-6))

            pred_total = self.pred_total[sem_cls]
            center = self.acc_translation_per_class[sem_cls]
            rotation = self.acc_rotation_per_class[sem_cls]
            scale = self.acc_scale_per_class[sem_cls]
            good_t_per_class[sem_cls]   = float(center / (pred_total + 1e-6))
            good_r_per_class[sem_cls]   = float(rotation / (pred_total + 1e-6))
            good_s_per_class[sem_cls]   = float(scale / (pred_total + 1e-6))

            eval_dict[cat_name] = [accuracy_per_class[sem_cls], prediction, n_total, center, rotation, scale, pred_total]

        # Mean scores
        class_mean_accuracy     = np.mean([v for k,v in accuracy_per_class.items()])
        class_mean_translation  = np.mean([v for k,v in good_t_per_class.items()])
        class_mean_rotation     = np.mean([v for k,v in good_r_per_class.items()])
        class_mean_scale        = np.mean([v for k,v in good_s_per_class.items()])

        return instance_mean_accuracy, class_mean_accuracy, class_mean_translation, class_mean_rotation, class_mean_scale, eval_dict