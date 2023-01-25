#!/usr/bin/env python3


import torch
import torch.nn as nn


def _make_layers(in_channels, config, batch_norm=True):
    _layers = []
    _in_ch = in_channels
    for _cfg in config:
        if _cfg == 'M':
            _layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if batch_norm:
                _layers += [nn.Conv2d(_in_ch, _cfg, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(_cfg),
                            nn.ReLU(inplace=True)]
            else:
                _layers += [nn.Conv2d(_in_ch, _cfg, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True)]
            _in_ch = _cfg
    return nn.Sequential(*_layers)


class UNet(nn.Module):

    def __init__(self, in_channels, n_class):
        super().__init__()

        self.in_channels = in_channels
        self.n_class = n_class

        # encoder
        self.enc1 = _make_layers(self.in_channels, [64, 64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _make_layers(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = _make_layers(128, [256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = _make_layers(256, [512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # center
        self.center = _make_layers(512, [1024, 1024])

        # decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _make_layers(1024, [512, 512])
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _make_layers(512, [256, 256])
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _make_layers(256, [128, 128])
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _make_layers(128, [64, 64])

        # output
        self.score = nn.Conv2d(64, self.n_class, kernel_size=1, stride=1, padding=0)

        # initialize weights
        self._initialize_weight()

    def forward(self, x):

        # encoder
        e1 = self.enc1(x)
        h = self.pool1(e1)
        e2 = self.enc2(h)
        h = self.pool2(e2)
        e3 = self.enc3(h)
        h = self.pool3(e3)
        e4 = self.enc4(h)
        h = self.pool4(e4)

        # center
        h = self.center(h)

        # decoder
        h = self.up4(h)
        h = self.dec4(torch.cat([e4, h], 1))
        h = self.up3(h)
        h = self.dec3(torch.cat([e3, h], 1))
        h = self.up2(h)
        h = self.dec2(torch.cat([e2, h], 1))
        h = self.up1(h)
        h = self.dec1(torch.cat([e1, h], 1))

        # output
        h = self.score(h)

        return h

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':

    unet = UNet(in_channels=3, n_class=5)
