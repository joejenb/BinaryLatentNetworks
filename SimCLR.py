import torch
from torch import nn
import torchvision
import copy

from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad

from utils import straight_through_round
from models.HopVAE import HopVAE

class SimCLR(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.backbone = HopVAE(config, device)
        self.projection_head = SimCLRProjectionHead(config.embedding_dim * (config.representation_dim ** 2), 512, config.num_features)
        self.classification_head = SimCLRProjectionHead(config.num_features, config.num_features // 2, config.num_classes)

    def forward(self, x):
        x, z = self.backbone(x)
        z = self.projection_head(z.flatten(start_dim=1))
        c = self.classification_head(z.detach())
        return c, z, x