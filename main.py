import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchmetrics.functional.classification import accuracy

import argparse

import numpy as np
import os

import wandb

from SimCLR import SimCLR

from utils import get_data_loaders, load_from_checkpoint, MakeConfig

from configs.cifar10_32_config import config

from lightly.loss import NTXentLoss

wandb.init(project="Binary-SimCLR", config=config)
config = MakeConfig(config)

def train(model, config, train_loader, optimiser, scheduler):

    model.train()
    train_error = 0
    train_accuracy = 0
    
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
    ntx_ent_loss = NTXentLoss()

    for (x0, x1), t, _ in train_loader:

        x0 = x0.to(model.device)
        x1 = x1.to(model.device)
        t = t.to(model.device)

        c0, z0 = model(x0)
        c1, z1 = model(x1)

        loss = ntx_ent_loss(z0, z1) + cross_entropy_loss(c0, t)

        train_error += loss.detach()

        train_accuracy += 0.5 * accuracy(c0, t, task="multiclass", num_classes=config.num_classes)
        train_accuracy += 0.5 * accuracy(c1, t, task="multiclass", num_classes=config.num_classes)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    scheduler.step()
    print({
        "Train Error": train_error / len(train_loader),
        "Train Accuracy": train_accuracy / len(train_loader)
    }, "\n")


def test(model, config, test_loader):
    # Recall Memory
    model.eval() 
    test_error = 0
    test_accuracy = 0

    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    with torch.no_grad():

        for x, t, _ in test_loader:

            x = x.to(model.device)
            t = t.to(model.device)

            c, _ = model(x)

            loss = cross_entropy_loss(c, t)

            test_error += loss.detach()

            test_accuracy += accuracy(c, t, task="multiclass", num_classes=config.num_classes)


    print({
            "Test Error": test_error / len(test_loader),
            "Test Accuracy": (test_accuracy) / len(test_loader)
        }, "\n")


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

    model = SimCLR(num_features=config.num_features, num_classes=config.num_classes, device=device).to(device)
    model = load_from_checkpoint(model, checkpoint_location)

    optimiser = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=config.gamma)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        train(model, config, train_loader, optimiser, scheduler)

        if not epoch % 5:
            test(model, config, test_loader)

        if not epoch % 5:
            torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()