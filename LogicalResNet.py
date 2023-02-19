import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

import numpy as np

from utils import straight_through_round

class LogicalResNet(ResNet):
    
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], tree_depth=10, num_features=10000, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, device="cpu"):
        super(LogicalResNet, self).__init__(block, layers, num_features, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation, norm_layer)

        self.device = device
        logical_body = [nn.Linear(num_features / (depth+1), num_features / (depth+2), bias=False) for depth in range(tree_depth - 1)]
        logical_head = [nn.Linear(num_features / tree_depth, num_classes, bias=False)]
        self.logical_tree = nn.ModuleList(logical_body + logical_head)
        
    def forward(self, x):
        features = self._forward_impl(x)
        binary_features = features

        output = binary_features
        for formulas in self.logical_tree[:-1]:
            output = F.relu(formulas(output))
        
        return self.logical_tree[-1](output)


