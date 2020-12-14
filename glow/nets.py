import numpy as np
import torch
import torch.nn as nn

from .utils import Conv2dZeros, Graph, st_gcn

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
    
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
    
    def init_hidden(self):
        self.do_init = True