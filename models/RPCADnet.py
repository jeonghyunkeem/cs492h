# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from models.cad_proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper_rpcad import get_loss
        
class RPCADNet(nn.Module):
    def __init__(self, num_class, num_heading_bin,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_proposal, sampling)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        point_cloud = inputs['point_clouds']
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # --------- BOUNDING BOX & CAD ALIGNMENT PROPOSAL ---------
        end_points = self.pnet(xyz, features, point_cloud, end_points)

        return end_points


# if __name__=='__main__':
#     sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
#     from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
#     from loss_helper import get_loss

#     # Define model
#     model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
#     try:
#         # Define dataset
#         TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

#         # Model forward pass
#         sample = TRAIN_DATASET[5]
#         inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
#     except:
#         print('Dataset has not been prepared. Use a random sample.')
#         inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

#     end_points = model(inputs)
#     for key in end_points:
#         print(key, end_points[key])

#     try:
#         # Compute loss
#         for key in sample:
#             end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
#         loss, end_points = get_loss(end_points, DC)
#         print('loss', loss)
#         end_points['point_clouds'] = inputs['point_clouds']
#         end_points['pred_mask'] = np.ones((1,128))
#         dump_results(end_points, 'tmp', DC)
#     except:
#         print('Dataset has not been prepared. Skip loss and dump.')
