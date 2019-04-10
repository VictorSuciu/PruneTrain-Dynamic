"""
Flattened VGG8 for CIFAR
"""

import torch.nn as nn
import math

__all__ = ['vgg8_bn_flat']

class VGG8(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=10):
        super(VGG8, self).__init__()
        self.conv1  = nn.Conv2d(3, 22, kernel_size=3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(22)
        # MaxPool
        self.conv2  = nn.Conv2d(22, 65, kernel_size=3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(65)
        # MaxPool
        self.conv3  = nn.Conv2d(65, 130, kernel_size=3, padding=1, bias=False)
        self.bn3    = nn.BatchNorm2d(130)
        # MaxPool
        self.conv4  = nn.Conv2d(130, 142, kernel_size=3, padding=1, bias=False)
        self.bn4    = nn.BatchNorm2d(142)
        # MaxPool
        self.conv5 = nn.Conv2d(142, 51, kernel_size=3, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(51)
        # MaxPool
        self.fc     = nn.Linear(51, num_classes)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu   = nn.ReLU(inplace=True)

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # This part of architecture remains the same
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def vgg8_bn_flat(**kwargs):
    model = VGG8(**kwargs)
    return model
