import snntorch as snn
from snntorch import surrogate
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Callable
import visualize
import dataset as ds
import SNN_func as snnfn


nCublets = 1000
nSensors = 100
max_t = 20
dt = 0.02
timesteps = int(max_t/dt)


labels_map = {
  -1: "unclassified",
   0: "proton",
   1: "kaon",
   2: "pion",
   3: "other"
}

nClasses = len(labels_map)-1


# Network Descriptions

population = 20

net_desc = {
    "layers" : [400, 120, 120, population],
    "timesteps": 100,
    "neuron_params" : {
                1: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],
                2: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],
                3: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True, 
                    "spike_grad": surrogate.atan(),
                    }]
                }
    }

#net_desc_membrane = deepcopy(net_desc)
#net_desc_membrane["output"] = "membrane"

net_desc_spikefreq = deepcopy(net_desc)
net_desc_spikefreq["output"] = "spike"

def _spikegen(data):
    spike_data = data.transpose(0,1) 
    spike_train = torch.where(spike_data > 300, torch.tensor(1), torch.tensor(0)).to(torch.float32) 
    return spike_train

def spikegen_multi(data, multiplicity=4):
    og_shape = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity*og_shape[2])
    for i in range(multiplicity):
        condition = data > np.power(10, i+2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity*sensor_idx+i] = 1

    return spike_data

def predict_spikefreq(output):
    prediction = output.sum(0).mean(1) # sum spikes across time and average over the population
    return prediction

def distance(prediction, targets, absolute: bool = True, relative: bool = True,
             transform: Callable = lambda *args, **kwargs: args[0]):
    p, t = transform(prediction), transform(targets)
    accuracy = t - p
    if absolute:
        accuracy = torch.abs(accuracy)
    if relative:
        accuracy /= t
    return accuracy


dataset = ds.build_dataset("../Data/PrimaryOnly/Uniform", max_files=100, primary_only=True,
                           target=["particle"])
train_dataset, test_dataset, val_dataset = ds.build_loaders(dataset, (0.7, 0.15), shuffle=True, num_workers=0)


batch_size = 50
nw = 0
train_load = DataLoader(data_train[:(len(data_train)//batch_size)*batch_size], batch_size=batch_size, shuffle=True, num_workers=nw)
val_load   = DataLoader(data_val[:(len(data_val)//batch_size)*batch_size],     batch_size=batch_size, shuffle=True, num_workers=nw)
test_load  = DataLoader(data_test[:(len(data_test)//batch_size)*batch_size],   batch_size=batch_size, shuffle=True, num_workers=nw)


net_Epos_spk = snnfn.Spiking_Net(net_desc_spikefreq, lambda x: spikegen_multi(x,4)) 
Pred_Epos_spk = snnfn.Predictor(predict_spikefreq, distance, population_sizes=20)
loss_Epos = snnfn.multi_MSELoss(weights=torch.tensor([1,1,1,1]))
opt_Epos_spk = torch.optim.Adam(net_Epos_spk.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=1e-3)
sche_Epos_spk = torch.optim.lr_scheduler.ExponentialLR(opt_Epos_spk, gamma=0.7)
train_Epos_spk = snnfn.Trainer(net_Epos_spk, loss_Epos, opt_Epos_spk, Pred_Epos_spk,
                    train_dataset, val_dataset, test_dataset)

num_epochs = 5
train_Epos_spk.train(num_epochs)
train_Epos_spk.predict.accuracy_fn = lambda p, t: distance(p, t, absolute=True, relative=True)
train_Epos_spk.test("test")
train_Epos_spk.predict.accuracy_fn = lambda p, t: distance(p, t, absolute=False, relative=False)
train_Epos_spk.show_results(nbins=50, title=["particle class"])




