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
        logical_body = [nn.Linear(num_features // (depth+1), num_features // (depth+2), bias=False) for depth in range(tree_depth - 1)]
        logical_head = [nn.Linear(num_features // tree_depth, num_classes, bias=False)]
        self.logical_tree = nn.ModuleList(logical_body + logical_head)
        
        for formulas in self.logical_tree:
            torch.nn.init.normal_(formulas.weight, mean=0.5, std=0.5)
        
    def forward(self, x):
        features = F.relu(self._forward_impl(x))
        binary_features = straight_through_round(features)

        output = binary_features
        # Want to straight_through_round weights and then use cloned output in multiplication
        # Want to fire only if formula evaluates to true -> should sum to number of out features - 1
        # Out features is given by first dimension of weights matrix 
        for formulas in self.logical_tree[:-1]:
            rounded_weights = straight_through_round(formulas.weight)
            output = F.linear(output, rounded_weights)# - rounded_weights.size(dim=0) // 2
            print(output)
            output = straight_through_round(F.sigmoid(output))
            print(output)
            print(formulas.weight)
            print(rounded_weights)
            print("\n")
        
        return self.logical_tree[-1](output)


