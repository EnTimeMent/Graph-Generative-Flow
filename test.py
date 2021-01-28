from glow.modules import ActNorm, InvertibleConv1x1, AffineCoupling
from glow.models import FlowStep, Block, Glow
import torch
from configs.utils import JsonConfig
import os
import argparse

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--cfg", default='configs/locomotion.json', type=str, help="config file")

from numpy.lib.stride_tricks import as_strided
import numpy as np
from glow.nets import Multi_LSTMs

if __name__ == "__main__":
    # test each module works 
    # x0 = torch.randn([1, 4, 21, 20])
    
    # AC = ActNorm(num_channels=3, scale=1)
    # IC = InvertibleConv1x1(3)
    # ST = AffineCoupling(in_channels=3, hidden_size=512)

    # y1, logdet = AC(x0)
    # y2, logdet = IC(y1, logdet)
    # y3, logdet = ST(y2, logdet)
    
    # x1 = ST(y3, reverse=True)
    # x2 = IC(x1, reverse=True)
    # x3 = AC(x2, logdet=None, reverse=True)
    
    # # test flows
    # x0 = torch.randn([1, 4, 21, 20])
    # FS = FlowStep(3, 64)
    # y1, logdet = FS(x0)
    # x1 = FS(y1, reverse=True)
    
    # # test blocks
    # x0 = torch.randn([1, 4, 21, 20])
    # B = Block(3, 64, 4, split=True)
    # logdet, logp, out, z = B(x0)
    # x1 = B(out, z, reverse=True)
    
    # test glows
    x0 = torch.randn([2, 3, 21, 4])
    cond0 = torch.randn([2, 3, 4])
    args = parser.parse_args()
    cfg = JsonConfig(args.cfg)
    model = Glow(cfg)
    logp, logdet, zs, loss = model(x0, cond0)
    
    y = model(zs, cond0, reverse=True)
    print('')
    
    
    # test sample
    x1 = torch.randn([2, 3, 21, 16])
    z_shape = x1.shape
    z = torch.normal(mean=torch.rand(z_shape),
                           std=torch.rand(z_shape) * 1)
    
    print('')
    
    # test LSTM
    # x0 = torch.randn([1, 3, 21, 16])
    lstm =  Multi_LSTMs(3, 21, 3)
    lstm.init_hidden()
    zss, normals = lstm(zs)
    zsss, normalss = lstm(zss)
    
    print('')