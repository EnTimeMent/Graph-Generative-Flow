import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

def calc_z_shapes(n_channel, t_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        t_size //= 2
        z_shapes.append((n_channel, 21, t_size))

    t_size //= 2
    z_shapes.append((n_channel * 2, 21, t_size))

    return z_shapes

class Trainer(object):
    def __init__(self, model, optim, schedule, data, logdir, device, cfg):
        
        self.cfg = cfg
        self.log_dir = logdir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.plot_dir = os.path.join(self.log_dir, "plots")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        cfg.dump(self.log_dir)
        self.checkpoints_gap = cfg.Train.check_gap
        
        self.model = model
        self.device = device
        self.optim = optim
        self.schedule = schedule
        self.init_lr = cfg.Optim.lr

        self.max_grad_clip = cfg.Train.max_grad_clip
        self.max_grad_norm = cfg.Train.max_grad_norm

        self.num_epochs = cfg.Train.num_epochs
        self.batch_size = cfg.Train.batch_size
        self.global_step = 0

        self.data = data
        self.train_dataset = data.train_dataset
        self.val_dataset = data.validation_dataset

        self.train_data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=cfg.Train.num_workers,
                                      shuffle=cfg.Data.shuffle,
                                      drop_last=True)
        self.val_data_loader = DataLoader(self.val_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=cfg.Train.num_workers,
                                          drop_last=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = cfg.Train.scalar_log_gap
        self.scalar_log_gaps = cfg.Train.scalar_log_gap
        self.validation_log_gaps = cfg.Train.validation_log_gap
        self.test_log_gaps = cfg.Train.test_log_gap

    def train(self):
        # begin to train
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            progress = tqdm(self.train_data_loader, desc='Batchs')
            
            for i_batch, batch in enumerate(progress):
                
                # set to training stata
                self.model.train()
                
                # update learning rate
                lr = self.schedule(init_lr=self.init_lr, global_step=self.global_step)
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0 and self.global_step > 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)

                # get batch data
                x = batch["joints"].permute(4, 0, 1, 2, 3).to(self.device)
                c = batch["controls"].permute(3, 0, 1, 2).to(self.device)
             
                # at first time, initialize ActNorm
                if self.global_step == 0:
                    _, _, zs, loss = self.model(x[0], c[0])
                    self.global_step += 1                    
                                
                glow_loss_sequence = 0
                nn_loss_sequence = 0
                total_loss_sequence = 0
                zs_history = []
                
                for i_step in range(len(x)): 
                    # forward phase
                    _, _, zs, glow_loss = self.model(x[i_step], c[i_step])
                                        
                    if i_step == 0:
                        zs_history.append(zs)
                        self.model.module.multi_lstms.init_hidden()
                        continue
                    
                    zs_pred, normals = self.model.module.prior(zs_history)
                    zs_history.append(zs)
                    
                    nn_loss = self.model.module.nn_loss(normals, zs)
                    total_loss = glow_loss + nn_loss

                    glow_loss_sequence += glow_loss
                    nn_loss_sequence += nn_loss
                    total_loss_sequence += total_loss

                    # backward
                    self.model.zero_grad()
                    self.optim.zero_grad()
                                        
                    total_loss.backward()
                    
                    # operate grad
                    if self.max_grad_clip is not None and self.max_grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        if self.global_step % self.scalar_log_gaps == 0:
                            self.writer.add_scalar("grad_norm", grad_norm, self.global_step)
                    
                    # step
                    self.optim.step()

                # loss
                total_loss_sequence /= len(x)
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("total_loss", total_loss_sequence, self.global_step)
                
                # save checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    state = {
                        "global_step": self.global_step,
                        # DataParallel wrap model in attr `module`.
                        "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                    }
                    _file_at_step = "save_{}k{}.pkg".format(int(self.global_step // 1000), int(self.global_step % 1000))
                    save_path = os.path.join(self.checkpoints_dir, _file_at_step)
                    torch.save(state, save_path)

                # validation
                if self.global_step % self.validation_log_gaps == 0:    
                    loss_val = 0
                    n_batches = 0
                    
                    self.model.eval() 
            
                    for i_val_batch, val_batch in enumerate(self.val_data_loader):
                        
                        # get batch data
                        x = val_batch["joints"].permute(4, 0, 1, 2, 3).to(self.device)
                        c = val_batch["controls"].permute(3, 0, 1, 2).to(self.device)

                        zs_history = []
                        
                        for i_step in range(len(x)): 
                            with torch.no_grad():
                                
                                _, _, zs, glow_loss = self.model(x[i_step], c[i_step])
                                if i_step == 0:
                                    zs_history.append(zs)
                                    self.model.module.multi_lstms.init_hidden()
                                    continue
                                
                                zs_pred, normals = self.model.module.prior(zs_history)
                                zs_history.append(zs)
                                nn_loss = self.model.module.nn_loss(normals, zs)
                                total_loss = glow_loss + nn_loss
                                loss_val += (total_loss / len(x))
                                           
                        n_batches += 1
                    loss_val /= n_batches  
                    self.writer.add_scalar("val_loss", loss_val, self.global_step)

                # test samples generation
                if self.global_step % self.test_log_gaps == 0:
                    z_sample = []
                    z_shapes = calc_z_shapes(3, self.cfg.Data.seqlen, self.cfg.Glow.L)

                    for z in z_shapes:
                        z_new = torch.randn(1, *z)
                        z_sample.append(z_new.to(self.device))
                                        
                self.global_step += 1 
                
            