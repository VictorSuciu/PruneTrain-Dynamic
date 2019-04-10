"""
Flattened VGG16 for CIFAR
"""

import torch.nn as nn
import math

__all__ = ['vgg13_bn_flat']

class VGG13(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=10):
        super(VGG13, self).__init__()
        self.conv1  = nn.Conv2d(3, 13, kernel_size=3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(13)
        self.conv2  = nn.Conv2d(13, 43, kernel_size=3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(43)
        # MaxPool
        self.conv3  = nn.Conv2d(43, 79, kernel_size=3, padding=1, bias=False)
        self.bn3    = nn.BatchNorm2d(79)
        self.conv4  = nn.Conv2d(79, 105, kernel_size=3, padding=1, bias=False)
        self.bn4    = nn.BatchNorm2d(105)
        # MaxPool
        self.conv5  = nn.Conv2d(105, 142, kernel_size=3, padding=1, bias=False)
        self.bn5    = nn.BatchNorm2d(142)
        self.conv6  = nn.Conv2d(142, 133, kernel_size=3, padding=1, bias=False)
        self.bn6    = nn.BatchNorm2d(133)
        # MaxPool
        self.conv7  = nn.Conv2d(133, 50, kernel_size=3, padding=1, bias=False)
        self.bn7    = nn.BatchNorm2d(50)
        self.conv8  = nn.Conv2d(50, 30, kernel_size=3, padding=1, bias=False)
        self.bn8    = nn.BatchNorm2d(30)
        # MaxPool
        self.conv9  = nn.Conv2d(30, 18, kernel_size=3, padding=1, bias=False)
        self.bn9    = nn.BatchNorm2d(18)
        self.conv10 = nn.Conv2d(18, 17, kernel_size=3, padding=1, bias=False)
        self.bn10   = nn.BatchNorm2d(17)
        self.fc     = nn.Linear(17, num_classes)
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
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def vgg13_bn_flat(**kwargs):
    model = VGG13(**kwargs)
    return model
