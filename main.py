import os
import argparse
import numpy as np
import torch
import datetime

from tqdm import tqdm
from configs.utils import JsonConfig

from glow.models import Glow
from glow.optimizer import get_optim
from glow.scheduler import get_schedule
from glow.trainer import Trainer
from dataset.locomotion import Locomotion

parser = argparse.ArgumentParser(description="Graph-Glow trainer")
parser.add_argument("--cfg", default='configs/locomotion.json', type=str, help="config file")

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = JsonConfig(args.cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")

    log_dir = os.path.join(cfg.Train.log, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    model = Glow(cfg).to(device)
    optimizer = get_optim[cfg.Optim.name](model.parameters(), cfg)  
    scheduler = get_schedule[cfg.Schedule.name]
    
    data = Locomotion(cfg)
    trainer = Trainer(model=model,
                      optim=optimizer,
                      schedule=scheduler,
                      data=data,
                      logdir=log_dir,
                      device=device,
                      cfg=cfg)

    trainer.train()