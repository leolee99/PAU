import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TotalLoss(nn.Module):
    def __init__(self, args, tau=5):
        super(TotalLoss, self).__init__()
        self.args = args
        self.loss_ce = CrossEn()

        self.loss_uct = UncertaintyAwareLoss(tau)
        self.loss_var = VarianceLoss()
    
    def forward(self, i2t_sims, t2i_sims, ret, lambda_=0.01):
        ground_truth = torch.arange(i2t_sims.shape[0],dtype=torch.long).cuda()      
        sim_loss1 = self.loss_ce(i2t_sims)
        sim_loss2 = self.loss_ce(t2i_sims)        
        sim_loss = (sim_loss1 + sim_loss2) / 2

        uct_loss1 = self.loss_uct(i2t_sims, ret['v_alpha'])
        uct_loss2 = self.loss_uct(t2i_sims, ret['t_alpha'])
        uct_loss = (uct_loss1 + uct_loss2) / 2

        var_loss = self.loss_var(ret['tt_logits'], ret['vv_logits'])

        loss = sim_loss + self.args.uct_weight * uct_loss + self.args.var_weight * var_loss
        loss_set = {'sim_loss': sim_loss, 'uct_loss': self.args.uct_weight * uct_loss, 'var_loss': self.args.var_weight * var_loss}
        return loss, loss_set


class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class UncertaintyAwareLoss(nn.Module):
    """
    Compute UncertaintyAwareLoss
    """
    def __init__(self, tau=5):
        super(UncertaintyAwareLoss, self).__init__()
        self.tau = tau
        #self.mse = nn.MSELoss(reduction='mean')
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.relu = nn.ReLU(inplace=True)

    # sims (K,K)
    def forward(self, sims, sim_K, lambda_=0.00005):
        BS = sims.size(0)
        K = sim_K.size(1)
        mask = 1 - torch.eye(BS).cuda()
        soft_label = (sims * mask).mean(1, keepdim=True)

        E = torch.exp(sim_K / self.tau)
        alpha = E + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        U = K / S

        scale =  (1 - U).mean() / soft_label.mean()

        # ce loss
        loss = self.mse(1 - U, scale * soft_label)
        return loss


class VarianceLoss(nn.Module):
    """
    Compute Variance Loss
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

