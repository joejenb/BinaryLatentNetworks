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
        self.classification_head = SimCLRProjectionHead(num_features, num_features // 2, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        c = self.classification_head(z.detach())
        return c, z

class BYOL(nn.Module):
    def __init__(self, num_features=1000, num_classes=10, device='cpu'):
        super().__init__()

        self.device = device

        resnet = torchvision.models.resnet18(num_classes=num_features)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(num_features, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)
        self.classification_head = BYOLPredictionHead(num_features, num_features // 2, num_classes)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        #y = straight_through_round(self.backbone(x).flatten(start_dim=1))
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        c = self.classification_head(y.detach())
        return c, p

    def forward_momentum(self, x):
        #y = straight_through_round(self.backbone_momentum(x).flatten(start_dim=1))
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z