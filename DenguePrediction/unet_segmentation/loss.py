#!/usr/bin/env python3


import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    # Reference impl.: https://www.jeremyjordan.me/semantic-segmentation/#loss

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        if target.dtype != torch.float32:
            target = target.type(torch.float32)
        
        target = target.view(input.size())

        axes = tuple(range(1, len(input.shape)))
        numerator = 2.0 * torch.sum(input * target, dim=axes)
        denominator = torch.sum(input**2 + target**2, dim=axes)

        return torch.mean(1.0 - (numerator / (denominator + self.eps)))
