"""
Flattened ResNet20 V1 for CIFAR with Bottleneck Blocks
"""

import torch.nn as nn
import torch
import math

__all__ = ['resnet20_v2_flat']

class ResNet20(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)

        #1
        self.bn1    = nn.BatchNorm2d(16)
        self.conv2  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2    = nn.BatchNorm2d(16)
        self.conv3  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)

        #2
        self.bn3    = nn.BatchNorm2d(16)
        self.conv4  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn4    = nn.BatchNorm2d(16)
        self.conv5  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)

        #3
        self.bn5    = nn.BatchNorm2d(16)
        self.conv6  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn6    = nn.BatchNorm2d(16)
        self.conv7  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)

        #4 (Stage 2)
        self.bn7    = nn.BatchNorm2d(16)
        self.conv8  = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn8    = nn.BatchNorm2d(32)
        self.conv9  = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn9    = nn.BatchNorm2d(32)
        self.conv10 = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False, stride=2)

        #5
        self.bn10   = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn11   = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)

        #6
        self.bn12   = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn13   = nn.BatchNorm2d(32)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)

        #7 (Stage 3)
        self.bn14   = nn.BatchNorm2d(32)
        self.conv15 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn15   = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn16   = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False, stride=2)

        #8
        self.bn17   = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn18   = nn.BatchNorm2d(64)
        self.conv19 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)

        #9
        self.bn19   = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn20   = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)

        self.bn21   = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(8)
        self.fc     = nn.Linear(64, num_classes)
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
        _x = self.conv1(x)

        #1
        #__x = torch.index_select(_x, 1, torch.tensor([0, 1, 3, 4]).cuda())
        x = self.bn1(_x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        #__x = torch.full([_x.size()[0], _x.size()[2], _x.size()[3]], 0).cuda()
        #x = torch.stack([x[:,0,:,:], __x, x[:,1,:,:], __x, x[:,2,:,:], x[:,3,:,:], __x,__x,__x,__x,__x,__x,__x,__x,__x,__x], dim=1)
        _x = _x + x

        #2
        x = self.bn3(_x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        _x = _x + x

        #3
        x = self.bn5(_x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        _x = _x + x

        #4 (Stage 2)
        _x = self.bn7(_x)
        _x = self.relu(_x)
        x = self.conv8(_x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        _x = self.conv10(_x)
        _x = _x + x

        #5
        x = self.bn10(_x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv12(x)
        _x = _x + x

        #6
        x = self.bn12(_x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.conv14(x)
        _x = _x + x

        #7 (Stage 3)
        _x = self.bn14(_x)
        _x = self.relu(_x)
        x = self.conv15(_x)
        x = self.bn15(x)
        x = self.relu(x)
        x = self.conv16(x)
        x = self.bn16(x)
        _x = self.conv17(_x)
        _x = _x + x

        #8
        x = self.bn17(_x)
        x = self.relu(x)
        x = self.conv18(x)
        x = self.bn18(x)
        x = self.relu(x)
        x = self.conv19(x)
        _x = _x + x

        #9
        x = self.bn19(_x)
        x = self.relu(x)
        x = self.conv20(x)
        x = self.bn20(x)
        x = self.relu(x)
        x = self.conv21(x)
        _x = _x + x

        x = self.bn21(_x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet20_v2_flat(**kwargs):
    model = ResNet20(**kwargs)
    return model
