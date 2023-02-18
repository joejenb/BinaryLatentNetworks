import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import argparse

import numpy as np
import os

import wandb

from LogicalResNet import LogicalResNet

from utils import get_data_loaders, load_from_checkpoint, MakeConfig

from configs.cifar10_32_config import config

wandb.init(project="LogicalResNet", config=config)
config = MakeConfig(config)

def train(model, train_loader, optimiser, scheduler):

    model.train()
    train_error = 0
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    for X, label in train_loader:
        X = X.to(model.device)
        label = label.to(model.device)

        optimiser.zero_grad()

        pred_label = model(X)

        pred_error = cross_entropy_loss(pred_label, label)
        loss = pred_error

        loss.backward()
        optimiser.step()
        
        train_error += pred_error.item()

    scheduler.step()
    wandb.log({
        "Train Error": (train_error) / len(train_loader.dataset)
    })


def test(model, test_loader):
    # Recall Memory
    model.eval() 
    test_error = 0
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    with torch.no_grad():
        for X, label in test_loader:
            X = X.to(model.device)
            label = label.to(model.device)

            pred_label = model(X)

            pred_error = cross_entropy_loss(pred_label, label)
            loss = pred_error

            test_error += pred_error.item()

    wandb.log({
            "Test Error": test_error / len(test_loader.dataset)
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)

    args = parser.parse_args()
    PATH = args.data 

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader = get_data_loaders(config, PATH)
    checkpoint_location = f'checkpoints/{config.data_set}-{config.image_size}.ckpt'
    output_location = f'outputs/{config.data_set}-{config.image_size}.ckpt'

    model = LogicalResNet(tree_depth=config.tree_depth, num_features=config.num_features, num_classes=config.num_classes, device=device).to(device)
    model = load_from_checkpoint(model, checkpoint_location)

    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=config.gamma)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        train(model, train_loader, optimiser, scheduler)

        if not epoch % 5:
            test(model, test_loader)

        if not epoch % 5:
            torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()