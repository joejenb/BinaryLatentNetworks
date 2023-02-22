import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
from torchvision import transforms

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MakeConfig:
    def __init__(self, config):
        self.__dict__ = config

def load_from_checkpoint(model, checkpoint_location):
    if os.path.exists(checkpoint_location):
        pre_state_dict = torch.load(checkpoint_location, map_location=model.device)
        to_delete = []
        for key in pre_state_dict.keys():
            if key not in model.state_dict().keys():
                to_delete.append(key)
        for key in to_delete:
            del pre_state_dict[key]
        for key in model.state_dict().keys():
            if key not in pre_state_dict.keys():
                pre_state_dict[key] = model.state_dict()[key]
        model.load_state_dict(pre_state_dict)
    return model

def straight_through_round(X):
    forward_value = torch.round(X)
    out = X.clone()
    out.data = forward_value.data
    return out

def get_data_loaders(config, PATH):
    if config.data_set == "MNIST":
        train_set = torchvision.datasets.MNIST(root=PATH, train=True, download=True)
        val_set = torchvision.datasets.MNIST(root=PATH, train=False, download=True)
        test_set = torchvision.datasets.MNIST(root=PATH, train=False, download=True)
        config.data_variance = 1

    elif config.data_set == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True)
        val_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True)
        test_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True)
        config.data_variance = 1

    train_set = LightlyDataset.from_torch_dataset(train_set)
    val_set = LightlyDataset.from_torch_dataset(val_set, transform=torchvision.transforms.ToTensor())
    test_set = LightlyDataset.from_torch_dataset(test_set, transform=torchvision.transforms.ToTensor())

    collate_fn = SimCLRCollateFunction(input_size=32)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

        



