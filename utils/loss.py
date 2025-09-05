import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

'''
alpha=0.25: 平衡因子，用於調整正樣本的權重
gamma=2: 聚焦參數，用於調整簡單/困難樣本的權重
size_average=True: 是否對損失進行平均
ignore_index=255: 忽略的標籤值
weight=None: 各類別的權重
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255, weight=None):  
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            ignore_index=self.ignore_index,
            weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
def create_loss(loss_type="focal", temporal_loss="none", temporal_weight=0.5, **kwargs):
    base_loss = FocalLoss(**kwargs)
    return base_loss
