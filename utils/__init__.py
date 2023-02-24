import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
from torchvision import transforms

import random

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

def make_noughts_and_crosses(dataset):
    # Use image dim of 20, padding of 2 in make grid -> get 64 by 64
    ones_idx = dataset.targets==1
    zeros_idx = dataset.targets==0

    data = {0: dataset.data[zeros_idx], 1: dataset.data[ones_idx]}

    grid_data, grid_targets = [], []
    out_data, out_targets = [], []

    move_prob = 0.0
    for state_num in range(50000):
        if not state_num % 9:
            if state_num:
                grid_data = torch.stack(grid_data, dim=0)
                out_data.append(torch.make_grid(grid_data, nrow=3))
                out_targets.append(torch.Tensor(grid_targets))

            next_move = random.choice([0, 1])
            grid_data, grid_targets = [], []

        if not state_num % 4000:
            move_prob += 0.1
        
        make_move = random.choice([0, 1], k=1, weights=[1 - move_prob, move_prob])

        if make_move:
            grid_targets.extend([0, 1] if next_move else [1, 0])
            move_data = data[next_move]
            grid_data.append(move_data[random.randint(len(move_data))])
            next_move = (next_move + 1) % 2
        else:
            grid_targets.extend([0, 0])
            grid_data.append(torch.zeros(20, 20))
    
    dataset.data = torch.stack(out_data, dim=0)
    dataset.targets = torch.stack(out_targets, dim=0)
    return dataset


def get_data_loaders(config, PATH):
    # Want to get list of ones and list of zeros for each dataset
    # Can make new dataset by -> iterating over every item -> every 9 iterations want to randomly choose one or zero to be first move
    # For each of 9 iterations then randomly choose either filled or not filled -> if filled use 'next players' symbol and then update this variable
    # Every 1000 iterations increase the probability of making move by 0.1
    # Keep count of moves made -> if all filled at last point -> must be blank
    # For targets want to have 1 large tensor with 2 columns -> possible codings of (0, 0), (0, 1), (1, 0)
    # Then apply make grid to each of them at the end of each 9 for labels just want as one list
    if config.data_set == "MNIST":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(20),
                transforms.Normalize((0.1307,), (0.3081,))
            ])


        train_set = make_noughts_and_crosses(torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=transform))
        val_set = make_noughts_and_crosses(torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform))
        test_set = make_noughts_and_crosses(torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform))

    elif config.data_set == "CIFAR10":
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.image_size),
                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
            ])
        train_set = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

        



