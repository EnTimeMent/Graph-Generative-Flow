import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from .modules import ActNorm, InvertibleConv1x1, Permute, AffineCoupling
from .nets import Multi_LSTMs
from .utils import Conv2dZeros, split_feature, multi_scales

class FlowStep(nn.Module):
    def __init__(self, num_in_channels, 
                 hidden_size,
                 actnorm_scale=1.0, 
                 flow_permutation='invconv', 
                 flow_coupling='affine', 
                 net_type='gcn',
                 graph_scale=1.0,
                 layout='locomotion',
                 LU_decomposed=True):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.LU_decomposed = LU_decomposed
        # ActNorm
        self.actnorm = ActNorm(num_in_channels, scale=actnorm_scale)
        # Permutation
        self.invconv = InvertibleConv1x1(num_in_channels, LU_decomposed=True)
        # Affine Coupling
        self.affine_coupling = AffineCoupling(in_channels=num_in_channels, 
                                              hidden_size=hidden_size, 
                                              net_type=net_type,
                                              graph_scale=graph_scale,
                                              layout=layout,
                                              affine=True)

    def forward(self, x, cond, logdet=None, reverse=False):
        if not reverse:
            out, logdet = self.actnorm(x, logdet)
            out, logdet = self.invconv(out, logdet)
            out, logdet = self.affine_coupling(out, cond, logdet)
            return out, logdet
        else:
            out = self.affine_coupling(x, cond, reverse=reverse)
            out = self.invconv(out, reverse=reverse)
            out = self.actnorm(out, cond, reverse=reverse)
            return out


class Block(nn.Module):
    def __init__(self, in_channels, hidden_size, K, split,
                 actnorm_scale=1.0, 
                 flow_permutation='invconv', 
                 flow_coupling='affine', 
                 net_type='gcn',
                 graph_scale=1.0,
                 layout='locomotion',
                 LU_decomposed=True):
        super().__init__()
        
        self.flows = nn.ModuleList()
        for i in range(K):
            self.flows.append(
                FlowStep(num_in_channels=in_channels*2,
                         hidden_size=hidden_size,
                         actnorm_scale=actnorm_scale,
                         flow_permutation='invconv', 
                         flow_coupling='affine', 
                         net_type='gcn',
                         graph_scale=graph_scale,
                         layout='locomotion',
                         LU_decomposed=True)
                )

        self.split = split
        if split:
            self.prior = Conv2dZeros(in_channels, in_channels*2)
        else:
            self.prior = Conv2dZeros(in_channels*2, in_channels*4)
        
            
    def forward(self, x, cond, z=None, logdet=None, reverse=False):
        if not reverse:
            N, C, V, T = x.shape
            squeezed = x.view(N, C, V, T//2, 2)
            squeezed = squeezed.permute(0, 1, 4, 2, 3)
            x = squeezed.contiguous().view(N, C*2, V, T//2)

            # N, C_cond, T = cond.shape
            # squeezed = cond.view(N, C_cond, T//2, 2)
            # squeezed = squeezed.permute(0, 1, 3, 2)
            # cond = squeezed.contiguous().view(N, C_cond*2, T//2)            

            for flow in self.flows:
                x, logdet = flow(x, cond, logdet)                   
                z = x
            
            if self.split:
                out, z = z.chunk(2, 1)
                mean, logsd = self.prior(out).chunk(2, 1)
                logp = self.logp(z, mean, logsd)
            else:
                zero = torch.zeros_like(z)
                mean, logsd = self.prior(zero).chunk(2, 1)
                logp = self.logp(z, mean, logsd)
                out = z
            
            return logdet, logp, out, z
        else:
            if self.split:
                x = torch.cat([x, z], 1)
            else:
                x = z
            
            for flow in self.flows[::-1]:
                x = flow(x, cond, reverse=reverse)
            
            N, C, V, T = x.shape
            unsqueezed = x.view(N, C//2, 2, V, T)
            unsqueezed = unsqueezed.permute(0, 1, 3, 4, 2)
            x = unsqueezed.contiguous().view(N, C//2, V, T*2)
            
            return x
                
    @staticmethod
    def likelihood(x, mean, logsd):
        device = x.device
        log2PI = torch.log(2 * torch.tensor(math.pi)).to(device)
        return -0.5 * (log2PI -logsd + (x-mean)**2) / torch.exp(2*logsd)

    @staticmethod
    def logp(x, mean, logsd):
        likelihood = Block.likelihood(x, mean, logsd)
        return torch.sum(likelihood, dim=[1, 2, 3])
    

class Glow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.joints = cfg.Glow.joints
        self.length = cfg.Glow.length
        self.L = cfg.Glow.L
        self.K = cfg.Glow.K
        self.hidden_size = cfg.Glow.hidden_channels
        
        self.blocks = nn.ModuleList()
        self.in_channels = cfg.Glow.in_channels
        # self.multi_lstms = Multi_LSTMs(num_channels=self.in_channels,
        #                                num_joints=21,
        #                                L=self.L,
        #                                num_layers=2)
        
        for i in range(self.L):
            split = i < (self.L - 1)
            block = Block(in_channels=self.in_channels,
                          hidden_size=self.hidden_size,
                          K=self.K,
                          split=split,
                          actnorm_scale=cfg.Glow.actnorm_scale,
                          flow_permutation=cfg.Glow.flow_permutation,
                          flow_coupling=cfg.Glow.flow_coupling,
                          net_type=cfg.Glow.net_type,
                          graph_scale=cfg.Glow.graph_scale,
                          layout=cfg.Glow.layout,
                          LU_decomposed=cfg.Glow.LU_decomposed)
            self.blocks.append(block)
    
    # def prior(self, zs_history):
    #     # self.multi_lstms()
    #     length = len(zs_history)
    #     L = len(zs_history[0])
    #     zs_pred = []

    #     # for i_level in range(L):
    #     #     sequence = []
    #     #     for i_step in range(length):
    #     #         sequence.append(zs_history[i_step][i_level])
            
    #     zs = zs_history[-1]
    #     zs_pred, normals = self.multi_lstms(zs)
        
    #     return zs_pred, normals
    
    def forward(self, x, cond, logdet=None, reverse=False):
        if not reverse:
            conds = multi_scales(cond, self.L)
            zs = []
            logp_sum = 0
            logdet_sum = 0
            for i, block in enumerate(self.blocks):
                logdet, logp, x, z = block(x, conds[i], logdet=logdet)
                zs.append(z)
                logdet_sum = logdet_sum + logdet
                
                if logp is not None:
                    logp_sum = logp_sum + logp
            
            nll = self.negative_log_likelihood(logdet, logp_sum)
            glow_loss = self.generative_loss(nll)
            
            # zs_pred, normals = self.multi_lstms(zs)

            return logp_sum, logdet_sum, zs , glow_loss
        else:
            with torch.no_grad():
                conds = multi_scales(cond, self.L)
                for i, block in enumerate(self.blocks[::-1]):
                    if i == 0:
                        y = block(x=x[-1], cond=conds[-1], z=x[-1], reverse=reverse)
                    else:
                        y = block(x=y, cond=conds[-(i+1)], z=x[-(i+1)], reverse=reverse)
            return y

    def negative_log_likelihood(self, logdet, logp):
        logdet_factor = self.in_channels * self.joints * self.length
        
        objective = logdet + logp
        
        nll = (-objective) / float(np.log(2.0) * logdet_factor)
        return nll

    def generative_loss(self, nll):
        return torch.mean(nll)
    
    # def nn_loss(self, normals, zs):
    #     L = len(normals)
    #     sum_loss = 0
    #     for i in range(L):
    #         loss = normals[i].log_prob(zs[i])
    #         sum_loss += torch.mean(loss)
        
    #     return -sum_loss
