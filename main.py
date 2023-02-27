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

from utils import get_data_loaders, load_from_checkpoint, MakeConfig, KNNClassifier

from configs.cifar10_32_config import config

from lightly.loss import NTXentLoss

wandb.init(project="Binary-SimCLR", config=config)
config = MakeConfig(config)

def train(model, config, train_loader, optimiser, scheduler):

    model.train()
    train_error = 0
    log_dict = dict()
    
    ntx_ent_loss = NTXentLoss()

    for (x0, x1), t, _ in train_loader:

        x0 = x0.to(model.device)
        x1 = x1.to(model.device)
        t = t.to(model.device)

        _, z0 = model(x0)
        _, z1 = model(x1)

        loss = ntx_ent_loss(z0, z1)

        train_error += loss.detach()
        log_dict["Train Error"] = train_error / len(train_loader)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    scheduler.step()
    wandb.log(log_dict)

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

            c = model(x)

            loss = cross_entropy_loss(c, t)

            test_error += loss.detach()

            test_accuracy += accuracy(c, t, task="multiclass", num_classes=config.num_classes)


    wandb.log({
            "Test Error": test_error / len(test_loader),
            "Test Accuracy": (test_accuracy) / len(test_loader)
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

    model = SimCLR(num_features=config.num_features, num_classes=config.num_classes, device=device).to(device)
    model = load_from_checkpoint(model, checkpoint_location)

    knn_model = KNNClassifier(config.num_classes, device=model.device)

    optimiser = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, config.epochs)

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        train(model, config, train_loader, optimiser, scheduler)

        if not epoch % 5:
            knn_model.backbone = model.backbone
            knn_model.update_feature_bank(val_loader)
            test(knn_model, config, test_loader)

        if not epoch % 5:
            torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()