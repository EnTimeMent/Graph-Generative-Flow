import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .utils import split_feature
from .nets import STGCN

class ActNorm(nn.Module):
    def __init__(self, num_channels:int, scale:float=1.0):
        super().__init__()
        size = [1, num_channels, 1, 1] # N, C, V, T
        self.num_channels = num_channels
        self.inited = False
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        
    def _check_input_dim(self, x):
        assert len(x.size()) == 4
        assert x.size(1) == self.num_channels
    
    def initialize_parameters(self, x):
        if not self.training:
            return
        assert x.device == self.bias.device
        self.initialize_bias(x)
        self.initialize_logs(x)
        self.inited = True
    
    def initialize_bias(self, x):
        with torch.no_grad():
            bias = torch.mean(x, dim=[0, 2, 3], keepdim=True) * -1.0
            self.bias.data.copy_(bias.data)
        
    def initialize_logs(self, x):
        with torch.no_grad():
            std = torch.std(x, dim=[0, 2, 3], keepdim=True)
            scale = self.scale / (std+1e-6)
            logs = torch.log(scale)
            self.logs.data.copy_(logs.data)
    
    def _scale(self, x, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            y = x * torch.exp(logs)
        else:
            y = x * torch.exp(-logs)
        
        if logdet is not None:
            num_pixels = x.size(2) * x.size(3)
            dlogdet = torch.sum(logs) * num_pixels
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
            
        return y, logdet

    def _center(self, x, reverse=False):
        if not reverse:
            return x + self.bias
        else:
            return x - self.bias
    
    def forward(self, x, logdet=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(x)
        self._check_input_dim(x)
        
        if not reverse:
            x = self._center(x, reverse)
            y, logdet = self._scale(x, logdet, reverse)
            return y, logdet
        else:
            x, logdet = self._scale(x, logdet, reverse)
            y = self._center(x, reverse)
            return y


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True):
        super().__init__()
        self.num_channels = num_channels
        self.LU_decomposed = LU_decomposed
        weight_shape = [num_channels, num_channels]
        weight, _ = torch.qr(torch.randn(*weight_shape))
        
        if not self.LU_decomposed:
            self.weight = nn.Parameter(weight)
        else:
            weight_lu, pivots = torch.lu(weight)
            w_p, w_l, w_u = torch.lu_unpack(weight_lu, pivots)
            w_s = torch.diag(w_u)
            sign_s = torch.sign(w_s)
            log_s = torch.log(torch.abs(w_s))
            w_u = torch.triu(w_u, 1)
            
            u_mask = torch.triu(torch.ones_like(w_u), 1)
            l_mask = u_mask.T.contiguous()
            eye = torch.eye(l_mask.shape[0])
            
            self.register_buffer('p', w_p)
            self.register_buffer('sign_s', sign_s)
            self.register_buffer('eye', eye)
            self.register_buffer('u_mask', u_mask)
            self.register_buffer('l_mask', l_mask)
            self.l = nn.Parameter(w_l)
            self.u = nn.Parameter(w_u)
            self.log_s = nn.Parameter(log_s)
    
    def forward(self, x, logdet=None, reverse=False):
        num_pixels = x.size(2) * x.size(3)
        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * num_pixels
            if not reverse:
                weight = self.weight.unsqueeze(2).unsqueeze(3)
                y = F.conv1d(x, weight)
                if logdet is not None:
                    logdet = logdet + dlogdet
                return y, logdet
            else:
                weight = torch.inverse(self.weight.double()).float().unsqueeze(2).unsqueeze(3)
                y = F.conv1d(x, weight)
                if logdet is not None:
                    logdet = logdet - dlogdet
                return y
        else:
            l = self.l * self.l_mask + self.eye
            u = self.u * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = torch.sum(self.log_s) * num_pixels
            if not reverse:
                weight = torch.matmul(self.p, torch.matmul(l, u)).unsqueeze(2).unsqueeze(3)
                y = F.conv2d(x, weight)
                if logdet is not None:
                    logdet = logdet + dlogdet
                else:
                    logdet = dlogdet
                return y, logdet
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                weight = torch.matmul(u, torch.matmul(l, self.p.inverse())).unsqueeze(2).unsqueeze(3)
                y = F.conv2d(x, weight)
                if logdet is not None:
                    logdet = logdet - dlogdet
                else:
                    logdet = dlogdet
                return y


class Permute(nn.Module):
    def __init__(self, num_channels, shuffle=False):
        super().__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1,-1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        if shuffle:
            np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, x, reverse=False):
        assert len(x.size()) == 4
        if not reverse:
            return x[:, self.indices, :, :]
        else:
            return x[:, self.indices_inverse, :, :]
        

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_size=512, net_type='gcn', graph_scale=1.0, layout='locomotion', affine=True):
        super().__init__()
        self.affine = affine
        self.net_type = net_type
        if net_type == 'gcn':
            self.net = STGCN(input_dim=in_channels//2,
                           hidden_dim=hidden_size,
                           output_dim=2*(in_channels-in_channels//2),
                           layout=layout,
                           graph_scale=graph_scale)

    def forward(self, x, cond, logdet=None, reverse=False):
        if not reverse:
            cond1, cond2 = split_feature(cond, 'split')

            # step 1
            x1, x2 = split_feature(x, 'split')
            h = self.net(x1, cond1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.) + 1e-6
            x2 = (x2 + shift) * scale
            y = torch.cat([x1, x2], 1)
            
            # step 2
            # x1, x2 = split_feature(y, 'split')
            h = self.net(x2, cond2)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.) + 1e-6
            x1 = (x1 + shift) * scale
            y = torch.cat([x1, x2], 1)
            
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
            return y, logdet
        
        else:
            cond1, cond2 = split_feature(cond, 'split')
            
            # step 1
            x1, x2 = split_feature(x, "split")
            h = self.net(x2, cond2)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.) + 1e-6
            x1 = x1 / scale - shift
            # x = torch.cat([x1, x2], 1)
            
            # step 2
            # x1, x2 = split_feature(x, "split")
            h = self.net(x1, cond1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.) + 1e-6
            x2 = x2 / scale - shift
            y = torch.cat([x1, x2], 1)
            
            return y


if __name__ == "__main__":
    x0 = torch.randn([1, 3, 21, 20])
    
    AC = ActNorm(num_channels=3, scale=1)
    IC = InvertibleConv1x1(3)
    ST = AffineCoupling(in_channels=3, hidden_size=512)
    
    
    y1, logdet = AC(x0)
    y2, logdet = IC(y1, logdet)
    y3, logdet = ST(y2, logdet)
    
    x1 = ST(y3, reverse=True)
    x2 = IC(x1, reverse=True)
    x3 = AC(x2, logdet=None, reverse=True)
    
  
    print('')