import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .motion_data import MotionDataset

def mirror_data(data):
    aa = data.copy()
    aa[:,:,3:15]=data[:,:,15:27]
    aa[:,:,3:15:3]=-data[:,:,15:27:3]
    aa[:,:,15:27]=data[:,:,3:15]
    aa[:,:,15:27:3]=-data[:,:,3:15:3]
    aa[:,:,39:51]=data[:,:,51:63]
    aa[:,:,39:51:3]=-data[:,:,51:63:3]
    aa[:,:,51:63]=data[:,:,39:51]
    aa[:,:,51:63:3]=-data[:,:,39:51:3]
    aa[:,:,63]=-data[:,:,63]
    aa[:,:,65]=-data[:,:,65]
    return aa

def reverse_time(data):
    aa = data[:,-1::-1,:].copy()
    aa[:,:,63] = -aa[:,:,63]
    aa[:,:,64] = -aa[:,:,64]
    aa[:,:,65] = -aa[:,:,65]
    return aa

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled

def create_synth_test_data(n_frames, nFeats, scaler):

    syth_data = np.zeros((7,n_frames,nFeats))
    lo_vel = 1.0
    hi_vel = 2.5
    lo_r_vel = 0.08
    hi_r_vel = 0.08
    syth_data[0,:,63:65] = 0
    syth_data[1,:,63] = lo_vel
    syth_data[2,:,64] = lo_vel
    syth_data[3,:,64] = hi_vel
    syth_data[4,:,64] = -lo_vel
    syth_data[5,:,64] = lo_vel
    syth_data[5,:,65] = lo_r_vel
    syth_data[6,:,64] = hi_vel
    syth_data[6,:,65] = hi_r_vel
    syth_data = standardize(syth_data, scaler)
    syth_data[:,:,:63] = np.zeros((syth_data.shape[0],syth_data.shape[1],63))
    return syth_data.astype(np.float32)

class Locomotion():
    def __init__(self, cfg):
        data_root = cfg.Data.dir
        self.parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
        
        # load
        pbar = tqdm(range(4), desc='load dataset')
        train_series = np.load(os.path.join(data_root, 'all_locomotion_train_'+str(cfg.Data.framerate)+'fps.npz'))
        test_series = np.load(os.path.join(data_root, 'all_locomotion_test_'+str(cfg.Data.framerate)+'fps.npz'))
        train_data = train_series['clips'].astype(np.float32)                                           
        test_data = test_series['clips'].astype(np.float32)
        pbar.update()      
        
        # split
        pbar.set_description('split data')
        validation_data = train_data[-100:,:,:]
        train_data = train_data[:-100,:,:]
        pbar.update()
        
        # augmentation
        pbar.set_description('data augmentation')
        if cfg.Data.mirror:
            mirrored =  mirror_data(train_data)
            train_data = np.concatenate((train_data,mirrored), axis=0)
        if cfg.Data.reverse_time:
            rev = reverse_time(train_data)
            train_data = np.concatenate((train_data,rev), axis=0)
        pbar.update()
        
        # standardize
        pbar.set_description('data standardize')
        train_data, scaler = fit_and_standardize(train_data)
        validation_data = standardize(validation_data, scaler)
        test_data = standardize(test_data, scaler)
        pbar.update()
        pbar.close()
        
        # synthesize test data
        # --------
        synth_data = create_synth_test_data(test_data.shape[1], test_data.shape[2], scaler)
        all_test_data = np.concatenate((test_data, synth_data), axis=0)
        self.n_test = all_test_data.shape[0]
        n_tiles = (cfg.Train.batch_size // self.n_test) + 1
        all_test_data = np.tile(all_test_data.copy(), (n_tiles,1,1))
        
        self.scaler = scaler
        self.frame_rate = cfg.Data.framerate
        
        self.train_dataset = MotionDataset(train_data, cfg.Data.seqlen, cfg.Data.overlap)
        self.validation_dataset = MotionDataset(validation_data, cfg.Data.seqlen, cfg.Data.overlap)
        