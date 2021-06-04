# CS492H 2021 Prof. M Sung
# PA2 AutoEncoder
# Jeonghyun Kim

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetAE(nn.Module):
    def __init__ (self, latent=128, in_dims=3, n_points=2048):
        """
            Args:
                in_dims: int
                    input dimension(3 by default)
                n_points: int
                    number of points
                    
            1) Encoder: 4 layers
                    in_dims > 64 > 64 > 64 > 128 (> 1024)
            2) Decoder: 3 layers where n = n_points
                    128(or 1024) > n/2 > n > 3n
        """
        super(PointNetAE, self).__init__()
        
        self.in_dims = in_dims
        self.out_dims = in_dims
        self.latent = latent
        
        # Encoder
        self.conv1 = nn.Conv1d(self.in_dims, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Decoder
        self.fc1 = nn.Linear(128, int(n_points / 2))
        self.fc2 = nn.Linear(int(n_points / 2), n_points)
        self.fc3 = nn.Linear(n_points, n_points * 3)
        
        self.bn1_d = nn.BatchNorm1d(int(n_points / 2))
        self.bn2_d = nn.BatchNorm1d(n_points)
        
        if latent != 128:
            self.conv5 = nn.Conv1d(128, self.latent, 1)
            self.bn5 = nn.BatchNorm1d(self.latent)
            self.fc1 = nn.Linear(self.latent, int(n_points/2))
        
    def forward(self, x, r=False):
        """
            Args: 
                x: (B, N_points, in_dim)
            Returns: 
                recon_ret: (B, N_points, out_dim)
                latent: (B, latent_dim)
        """
        batch_size = x.shape[0] 
        n_points = x.shape[1]
        
        x = x.transpose(2, 1) # (B, in_dim, N)
        
        # Encoder
        e = F.relu(self.bn1(self.conv1(x)))
        e = F.relu(self.bn2(self.conv2(e)))
        e = F.relu(self.bn3(self.conv3(e)))
        e = F.relu(self.bn4(self.conv4(e)))
        if self.latent != 128:
            e = F.relu(self.bn5(self.conv5(e)))
            
        gf = torch.max(e, 2, keepdim=True)[0] # global feature

        latent = gf.reshape(batch_size, -1)
        
        if r:
            return latent

        # Decoder
        d = F.relu(self.bn1_d(self.fc1(latent)))
        d = F.relu(self.bn2_d(self.fc2(d)))
        d = self.fc3(d)
        
        recon_ret = d.reshape(batch_size, n_points, self.out_dims)
        
        return recon_ret, latent