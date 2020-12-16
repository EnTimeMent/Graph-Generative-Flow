import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .utils import Conv2dZeros, Graph, st_gcn, split_feature

class STGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 layout='locomotion', graph_scale=1,
                 edge_importance_weighting=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        graph = Graph(layout=layout, scale=graph_scale)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('A', A)
        
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.gcn_networks = nn.ModuleList((
            st_gcn(input_dim, hidden_dim, kernel_size, cond_res=True),
            st_gcn(hidden_dim, hidden_dim, kernel_size)
        ))
    
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for i in self.gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_networks)
            
        self.fcn = Conv2dZeros(hidden_dim, output_dim)
        
    def forward(self, x, cond):
        
        N, C, V, T = x.size() # input x: N, C, V, T
        x = x.permute(0, 1, 3, 2).contiguous() # N, C, T, V

        for gcn, importance in zip(self.gcn_networks, self.edge_importance):
            x = gcn(x, cond, self.A * importance)

        y = self.fcn(x).permute(0, 1, 3, 2)
        
        return y
    
class Multi_LSTMs(nn.Module):
    def __init__(self, num_channels, num_joints, L, num_layers=2):
        super(Multi_LSTMs, self).__init__()
        self.num_channels = num_channels
        self.num_joints = num_joints
        self.L = L
        
        self.LSTMs = nn.ModuleList()
        lstm_dim = num_channels * num_joints
        for l in range(L-1):
            self.lstm = nn.LSTM(lstm_dim, lstm_dim*2, num_layers, batch_first=True)
            self.LSTMs.append(self.lstm)

        self.lstm = nn.LSTM(lstm_dim*2, lstm_dim*4, num_layers, batch_first=True)
        self.LSTMs.append(self.lstm)    

    def init_hidden(self):
        self.do_init = True
        self.hiddens = []
    
    def prior(self, x):
        mean, logs = split_feature(x.detach(), "split")
        return mean, logs 

    def forward(self, xs):
        zs = []
        normals = []

        for i in range(self.L):
            x = xs[i]
            B, C, V, T = x.shape
            x = x.permute(0, 3, 1, 2).view(B, T, C*V).contiguous()
            if self.do_init:
                lstm_out, hidden = self.LSTMs[i](x)
            else:
                lstm_out, hidden = self.LSTMs[i](x, self.hiddens[i])
            lstm_out = lstm_out.view(B, T, 2*C, V).permute(0, 2, 3, 1)
            self.hiddens.append(hidden)

            mean, logs = self.prior(lstm_out)
            # z = torch.normal(mean=mean, std=torch.exp(logs))
            normal = Normal(loc=mean, scale=torch.exp(logs))
            z = normal.sample()
            zs.append(z)
            normals.append(normal)

        self.do_init = False
        return zs, normals
        