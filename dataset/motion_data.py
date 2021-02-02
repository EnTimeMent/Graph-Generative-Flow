import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, data, seqlen, overlap):
        self.seqlen = seqlen
        self.overlap = overlap  
        self.num_channels = 3
        self.num_joints = 21  
        joints = data[:, :, :-3]    # N, T, V*C
        self.num_sequence, self.num_frames, _ = joints.shape 
        
        joints = joints.reshape(self.num_sequence, self.num_frames, self.num_joints, self.num_channels)
        joints = joints.transpose(0, 3, 2, 1)
        controls = data[:, :, -3:]  # N, T, C    
        controls = controls.transpose(0, 2, 1)        

        self.joints = self.create_samples(joints).transpose(1, 2, 3, 4, 0)
        self.controls = self.create_samples(controls).transpose(1, 2, 3, 0)
        print('')
        
    def create_samples(self, data):
        samples = []
        i = 0
        while (i+self.seqlen) < self.num_frames:
            sample = data[:, ..., i:i+self.seqlen]
            samples.append(sample)
            i = i + self.overlap
        samples = np.array(samples)
        # samples = np.concatenate(samples, 0)
        return samples
    
    def __len__(self):
        return self.num_sequence
    
    def __getitem__(self, idx):
        sample = {
            'joints': self.joints[idx, ...],
            'controls': self.controls[idx, ...]
        }
        return sample
