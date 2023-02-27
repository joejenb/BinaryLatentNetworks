import torch
from torch import nn
import torchvision
import copy

from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad

from utils import straight_through_round

class SimCLR(nn.Module):
    def __init__(self, num_features=1000, num_classes=10, device='cpu'):
        super().__init__()
        resnet = torchvision.models.resnet18(num_classes=num_features)
        self.device = device
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 512, num_features)
        #self.classification_head = SimCLRProjectionHead(num_features, num_features // 2, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        c = None#self.classification_head(z.detach())
        return c, z