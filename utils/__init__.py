import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import lightly
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
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.image_size, config.image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

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
        val_set = LightlyDataset.from_torch_dataset(val_set, transform=test_transforms)
        test_set = LightlyDataset.from_torch_dataset(test_set, transform=test_transforms)

    collate_fn = SimCLRCollateFunction(
        input_size=config.image_size,
        vf_prob=0.5,
        rr_prob=0.5,
        cj_prob=0.0,
        random_gray_scale=0.0
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader


def knn_predict(feature: torch.Tensor,
                feature_bank: torch.Tensor,
                feature_labels: torch.Tensor, 
                num_classes: int,
                knn_k: int=200,
                knn_t: float=0.1) -> torch.Tensor:
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(
        feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(
        0) * knn_k, num_classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(
        0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class KNNClassifier(nn.Module):
    def __init__(self, num_classes, knn_k=200, knn_t=0.1, device='cpu'):
        super().__init__()

        self.backbone = nn.Module()
        self.num_classes = num_classes

        self.knn_k = knn_k
        self.knn_t = knn_t
        self.device = device


    def update_feature_bank(self, dataloader):
        
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []

        with torch.no_grad():
            for data in dataloader:
                x, target, _ = data
                x = x.to(self.device)
                target = target.to(self.device)

                feature = self.backbone(x).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)

        self.feature_bank = torch.cat(
            self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(
            self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def forward(self, input):
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            x, targets, _ = input

            feature = self.backbone(x).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature,
                self.feature_bank,
                self.targets_bank,
                self.num_classes,
                self.knn_k,
                self.knn_t
            )
            return pred_labels