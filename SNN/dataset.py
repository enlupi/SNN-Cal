import numpy as np
import math

import struct
import os

import snntorch as snn
import torch
import torch.nn as nn

import snntorch.spikeplot as splt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import MulticlassConfusionMatrix

from snntorch import surrogate
import snntorch.functional as SF
import torch.nn.functional as F
from torch.nn import NLLLoss, LogSoftmax

from snntorch import spikegen

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import snntorch.spikeplot as splt
import imageio

from sklearn.metrics import ConfusionMatrixDisplay

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###############################################################################


nCublets = 1000
nSensors = 100
max_t = 20
dt = 0.2
timesteps = int(max_t/dt)


###############################################################################


def readfile(filename, primary_only):

  ph_list = []
  E_list  = []
  sE_list = []
  N_list  = []
  p_class = []
  if not primary_only:
    primary_list = []

  with open(filename, 'rb') as file:

    data = file.read(4)
    while data:
    
      ph_matrix = np.zeros(shape=(timesteps, nSensors), dtype=np.int32)

      # read photon count data
      while True:
        t = struct.unpack('i', data)[0]
        # check stop condition
        if t == 2147483647:
          break
        sensor = struct.unpack('i', file.read(4))[0]
        n_photons = struct.unpack('i', file.read(4))[0]
        ph_matrix[t, sensor] = n_photons

        data = file.read(4)
      
      ph_list.append(ph_matrix)

      # Read cublet_id
      cublet_id = struct.unpack('i', file.read(4))[0]
      
      # Read total energy released
      E_list.append(struct.unpack('d', file.read(8))[0])
      
      # Read energy dispersion
      sE_list.append(struct.unpack('d', file.read(8))[0])
    
      # Read number of interactions
      N_list.append(struct.unpack('i', file.read(4))[0])
    
      # Read particle class
      p_class.append(struct.unpack('i', file.read(4))[0]-1)

      # Read primary vertex indicator
      if not primary_only:
        primary_list.append(struct.unpack('i', file.read(4))[0])

      data = file.read(4)

  res = [ph_list, E_list, sE_list, N_list, p_class]
  if not primary_only:
    res.append(primary_list)

  return res


###############################################################################


# converts to Torch tensor of desired type
def to_tensor_and_dtype(input, target_dtype=torch.float32):
    
    # Convert to PyTorch tensor
    if not torch.is_tensor(input):
        input = torch.tensor(input)

    # Force the tensor to have the specified dtype
    if input.dtype is not target_dtype:
        input = input.to(target_dtype)
    
    return input

class CustomDataset(Dataset):
    def __init__(self, filelist, primary_only=True, target="energy", transform=None):
        
        targets_dict = {
            "energy":1,
            "dispersion":2,
            "N_int":3,
            "particle":4
        }
        
        samples = []
        targets = []
        for file in filelist:
            info = readfile(file, primary_only)
            samples += info[0]
            targets += info[targets_dict[target]]

        samples = to_tensor_and_dtype(np.array(samples))
        #targets = F.one_hot(torch.tensor(targets)-1, nClasses)

        self.data = list(zip(samples, targets))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]  # type: ignore
        if isinstance(idx, (list, np.ndarray)):
            return [self.__getitem__(i) for i in idx]

        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

###############################################################################


def build_dataset(path, split=0.8, val=True, max_files=50, *args, **kwargs):
    
    filelist = []

    for subdir, _, files in os.walk(path):
        subdir_name = os.path.basename(subdir)
        if files:
            filelist += [os.path.join(path, subdir_name, f) for f in files[:max_files]]

    dataset = CustomDataset(filelist, *args, **kwargs)
    
    train_size = int(len(dataset)*split)
    test_size = len(dataset)-train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if val:
        val_size = int(train_size*(1-split))
        train_size -= val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        return train_dataset, test_dataset, val_dataset
    
    else:
        return train_dataset, test_dataset

