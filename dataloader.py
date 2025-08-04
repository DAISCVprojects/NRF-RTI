import os
import torch
import numpy as np
import torch.utils.data as data
from args import get_arguments
import h5py

args = get_arguments()

class RTI_data(data.Dataset):
    def __init__(self, 
                 root_dir,
                 transform = None,
                 mode = 'train'
                 ):
        self.mode = mode
        self.transform = transform
        self.root_dir = root_dir
        self.h5f = h5py.File(self. _get_file_location(self.mode.lower()), 'r')
    
    def __len__(self):
        num_entries = self.h5f['labels'].shape[0]
        return num_entries
    
    def __getitem__(self, index):
        feature = self.h5f['features'][index]
        label = self.h5f['labels'][index]
        
        feature = self.transform(feature)
        label = self.transform(np.array([label], dtype=np.float32))
        
        return feature, label
    
    def _get_file_location(self, filename):
        assert(filename in ['train', 'val', 'test'], "The mode should be 'train', 'val', 'test', but you wrote{0}".format(filename))
        name = filename + '.h5'
        source = os.path.join(self.root_dir, name)
        
        return source
    