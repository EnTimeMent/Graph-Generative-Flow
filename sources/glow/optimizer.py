import torch

def build_adam(params, cfg):
    return torch.optim.Adam(params,
                            lr=cfg.Optim.lr, 
                            betas=cfg.Optim.betas,
                            eps=cfg.Optim.eps)

get_optim = {
    "adam": build_adam
}