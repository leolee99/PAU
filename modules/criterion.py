from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import DropPath

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class UncertaintyAwareLoss(nn.Module):
    """
    Compute UncertaintyAwareLoss
    """
    def __init__(self, tau=5):
        super(UncertaintyAwareLoss, self).__init__()
        self.tau = tau
        self.mse = nn.MSELoss(reduce=True, size_average=True)
    
    # sims (K,K)
    def forward(self, sims, alpha, lambda_=0.01):
        BS = sims.size(0)
        K = alpha.size(1)
        mask = 1 - torch.eye(BS).cuda()
        soft_label = (sims * mask).mean(1, keepdim=True)
        S = torch.sum(alpha, dim=1, keepdim=True)
        U = K / S

        scale =  (1 - U).mean() / soft_label.mean()

        # ce loss
        loss = self.mse(1 - U, scale * soft_label)
        return loss


class VarianceLoss(nn.Module):
    """
    Compute UncertaintyAwareLoss
    """
    def __init__(self):
        super(VarianceLoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)

    # sims (K,K)
    def forward(self, vv, tt):
        K = vv.size(0)
        label = torch.zeros(vv.shape).cuda()
        mask = 1 - torch.eye(K).cuda()
        vv_m = mask * vv
        tt_m = mask * tt
        loss = self.mse(vv_m, label) + self.mse(tt_m, label)

        return loss

