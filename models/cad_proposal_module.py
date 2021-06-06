# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'scan2cad'))

RET_DIR = os.path.join(BASE_DIR, 'retrieval')
DUMP_DIR = os.path.join(RET_DIR, 'dump')

from pointnet2_modules import PointnetSAModuleVotes
from models.retrieval.autoencoder import PointNetAE
import pointnet2_utils
from CGNL import SpatialCGNL

import pickle
from scan2cad.s2c_config import Scan2CADDatasetConfig
import s2c_utils
DC = Scan2CADDatasetConfig()

def decode_scores(net, net2, end_points):
    # ----- Box Scores -----
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    box_size = net_transposed[:,:,5:8]
    end_points['box_size'] = box_size

    sem_cls_scores = net_transposed[:,:,8:] # Bxnum_proposalx21
    end_points['sem_cls_scores'] = sem_cls_scores

    # ----- Alignment Scores -----
    net2_transposed = net2.transpose(2,1)

    sym_scores = net2_transposed[:,:,:4]
    end_points['sym_scores'] = sym_scores

    scale_scores = net2_transposed[:,:,4:7]
    end_points['scale_scores'] = scale_scores

    rotation_scores = net2_transposed[:,:,7:]
    end_points['rot_6d_scores'] = rotation_scores # Bxnum_proposalx6
    # rotation_scores = net2_transposed[:,:,7:]
    # end_points['rotation_scores'] = rotation_scores # Bxnum_proposalx4

    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        # Object proposal/detection
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,
                                    2 + # Objectness
                                    3 + # Center Regression
                                    3 + # Box Size Regression
                                    self.num_class, # Num of classes
                                    1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        # CAD Retrieval
        # self.sem_clses = np.load(DUMP_DIR + '/all_category.npy')
        # self.filenames = np.load(DUMP_DIR + '/all_filenames.npy')
        # self.CADnet = PointNetAE(latent=512, in_dims=3, n_points=2048)
        # self.CADnet.load_state_dict(torch.load(os.path.join(BASE_DIR, 'retrieval/outputs') + '/model512_50.pth'))
        # self.CADnet.eval()    

        # Transform Prediction - from Box-Crop (Linear)
        # self.box_conv1 = torch.nn.Conv1d(3,64,1)
        # self.box_conv2 = torch.nn.Conv1d(64,64,1)
        # self.box_conv3 = torch.nn.Conv1d(64,128,1)
        # self.box_conv4 = torch.nn.Conv1d(128,128,1)
        # self.box_bn1 = torch.nn.BatchNorm1d(64)
        # self.box_bn2 = torch.nn.BatchNorm1d(64)
        # self.box_bn3 = torch.nn.BatchNorm1d(128)
        # self.box_bn4 = torch.nn.BatchNorm1d(128)

        # Transform Prediction - from Votes
        self.rot_param = 6
        self.cad_conv1 = torch.nn.Conv1d(128,128,1)
        self.cad_conv2 = torch.nn.Conv1d(128,128,1)
        self.cad_conv3 = torch.nn.Conv1d(128, 
                                        4 + # Symmetry
                                        3 + # Scale 
                                        self.rot_param,  # Rotation (6D)
                                        1)   
        self.cad_bn1 = torch.nn.BatchNorm1d(128)
        self.cad_bn2 = torch.nn.BatchNorm1d(128)

        # self-attention ========================================
        self.sa1_1 = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)
        self.sa2_1 = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)

        self.sa1_2 = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)
        self.sa2_2 = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)
        # =======================================================

    def BoxCropping(self, pcd, box_net):
        bboxes = box_net.transpose(2,1)     # (B, num_proposal, 2+3+3)
        K = bboxes.shape[1] # num_proposal
        B = bboxes.shape[0]

        batch_pc = pcd.cpu().numpy()[:,:,0:3]           # (B, N, 3)
        center = bboxes[:,:,2:5].detach().cpu().numpy() # (B, N, 3)
        size = bboxes[:,:,5:8].detach().cpu().numpy()   # (B, N, 3)

        # Output
        cad_inds = {}
        for b in range(B):
            pc = batch_pc[b,:,:]
            for k in range(K):
                box3d = s2c_utils.get_3d_box(size[b,k,:3], 0, center[b,k,:3])
                pc_in_box, inds = s2c_utils.extract_pc_in_box3d(pc, box3d)

                cad_inds[k] = np.where(inds == True)

                if len(pc_in_box) < 5:
                    k_feature = torch.zeros(64, 1).cuda()
                else:
                    # Points in Box
                    box_points = pcd[b, cad_inds[k], :3].transpose(2,1)     # (1, 3, inds)

                    # Alignment Feature
                    k_feature = self.box_conv1(box_points).squeeze(0)       # (64, inds) <- (1, 64, inds)
                    k_feature = torch.mean(k_feature, 1, keepdim=True)      # (64, 1)
            
                if k is 0:
                    B_feature = k_feature
                else:
                    B_feature = torch.cat((B_feature, k_feature), 1)    # (64, K)
            
            B_feature = B_feature.unsqueeze(0)  # (1, 64, K)

            if b is 0:
                feature = B_feature
            else:
                feature = torch.cat((feature, B_feature), 0)    # (B, 64, K)

        feature = F.relu(self.box_bn1(feature))     # (B, C, K)
        feature = F.relu(self.box_bn2(self.box_conv2(feature)))
        feature = F.relu(self.box_bn3(self.box_conv3(feature)))
        feature = F.relu(self.box_bn4(self.box_conv4(feature)))

        return feature

    # def CAD_Proposal(self, pcd, box_net, end_points, ret=False):
    #     bboxes = box_net.transpose(2,1)     # (B, num_proposal, 2+3+3)
    #     K = bboxes.shape[1] # num_proposal
    #     B = bboxes.shape[0]
    #     center = bboxes[:,:,2:5].detach().cpu().numpy()
    #     size = bboxes[:,:,5:8].detach().cpu().numpy()
    #     pred_crnrs_3d_upright_cam = np.zeros((B, K, 8, 3))
    #     for b in range(B):
    #         for k in range(K):
    #             crnrs_3d_upright_cam = s2c_utils.get_3d_box(size[b,k,:3], 0, center[b,k,:3])
    #             pred_crnrs_3d_upright_cam[b,k] = crnrs_3d_upright_cam

    #     # Retreival
    #     ret_cls     = np.zeros((B, K, 1), dtype=np.int64)
    #     ret_embed   = np.zeros((B, K, 512), dtype=np.float32)

    #     batch_pc = pcd.cpu().numpy()[:,:,0:3]   # (B, N, 3)
    #     if ret:
    #         with open(DUMP_DIR + '/shapenet_kdtree.pickle', 'rb') as pickle_file:
    #             database_kdtree = pickle.load(pickle_file)
    #             for b in range(B):
    #                 pc = batch_pc[b,:,:]
    #                 for k in range(K):
    #                     box3d = pred_crnrs_3d_upright_cam[b,k,:,:]
    #                     box3d = s2c_utils.flip_axis_to_depth(box3d)
    #                     pc_in_box, inds = s2c_utils.extract_pc_in_box3d(pc, box3d)
    #                     if len(pc_in_box) < 5:
    #                         continue
    #                     cad_inds = np.where(inds == True)
    #                     cad_pc = pcd[b, cad_inds, :3]
    #                     embedding = self.CADnet(cad_pc, r=True)
    #                     embedding = embedding.detach().cpu()
    #                     dist, pred_idx = database_kdtree.query(embedding, k=1)

    #                     # Output
    #                     category = self.sem_clses[pred_idx]

    #                     ret_cls[b, k, :] = category
    #                     ret_embed[b,k,:] = embedding
        
    #     end_points['sem_cls_scores'] = ret_cls
    #     end_points['embedding'] = ret_embed
    #     return end_points


    def forward(self, xyz, features, point_cloud, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
            point_cloud: (B, N, 3)
        Returns:
            scores: (B,num_proposal,-) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- LEARNING RELATION ---------
        # For CGNL
        feature_dim = features.shape[1]
        batch_size = features.shape[0]
        # features1 = features.contiguous().view(batch_size, feature_dim, 16, 16)
        
        # # --------- BOX PROPOSAL GENERATION ---------
        # votes relation
        # features1 = self.sa1_1(features1)
        # features1 = self.sa2_1(features1)
        # features1 = features1.contiguous().view(batch_size, feature_dim, self.num_proposal)
        # Box proposal
        net1 = F.relu(self.bn1(self.conv1(features))) 
        net1 = F.relu(self.bn2(self.conv2(net1))) 
        net1 = self.conv3(net1) # (batch_size, objness(2)+center(3)+size(3)+num_class, num_proposal)
        
        # --------- CAD ALGINMENT ESTIMATION ---------
        # Points Cropping by Box
        # features2 = self.BoxCropping(point_cloud, net1)
        # features2 = features.contiguous().view(batch_size, feature_dim, 16, 16)

        # Alignment Relation
        # features2 = self.sa1_2(features2)
        # features2 = self.sa2_2(features2)
        # features2 = features2.contiguous().view(batch_size, feature_dim, self.num_proposal)
        # Alignment Estimation
        net2 = F.relu(self.cad_bn1(self.cad_conv1(features)))
        net2 = F.relu(self.cad_bn2(self.cad_conv2(net2)))
        net2 = self.cad_conv3(net2) # (batch_size, symmetry(4)+scale(3)+rotation(6), num_proposal)

        # 6D Representation
        if self.rot_param > 4:
            rot_6d = net2.transpose(2,1)[:, :, 7:].contiguous() # (B, K, 6)
            e1 = F.normalize(rot_6d[:,:,:3], p=2, dim=-1)
            w = torch.bmm(e1.view(-1, 1,3), rot_6d[:,:,3:].view(-1,3,1)).view(rot_6d.shape[0],rot_6d.shape[1],1)
            e2 = F.normalize(rot_6d[:,:,3:] - w * e1, p=2, dim=-1)
            net2[:,7:,:] = torch.cat((e1, e2), dim=2).transpose(1,2)

        # --------- DECODE SCORES ---------
        end_points = decode_scores(net1, net2, end_points=end_points)
  
        return end_points


# if __name__=='__main__':
#     sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
#     from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
#     net = ProposalModule(DC.num_class, DC.num_heading_bin,
#         DC.num_size_cluster, DC.mean_size_arr,
#         128, 'seed_fps').cuda()
#     end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
#     out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
#     for key in out:
#         print(key, out[key].shape)
