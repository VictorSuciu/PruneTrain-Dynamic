"""
Flattened ResNet20 V1 for CIFAR with Bottleneck Blocks
"""

import torch.nn as nn
import math

__all__ = ['resnet32_v2_flat']

class ResNet32(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=10):
        super(ResNet32, self).__init__()
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

        #4
        self.bn7    = nn.BatchNorm2d(16)
        self.conv8  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn8    = nn.BatchNorm2d(16)
        self.conv9  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)

        #5
        self.bn9    = nn.BatchNorm2d(16)
        self.conv10  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn10    = nn.BatchNorm2d(16)
        self.conv11  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)

        #6 (Stage 2)
        self.bn11    = nn.BatchNorm2d(16)
        self.conv12  = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn12    = nn.BatchNorm2d(32)
        self.conv13  = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn13    = nn.BatchNorm2d(32)
        self.conv14 = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False, stride=2)

        #7
        self.bn14   = nn.BatchNorm2d(32)
        self.conv15 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn15   = nn.BatchNorm2d(32)
        self.conv16 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)

        #8
        self.bn16   = nn.BatchNorm2d(32)
        self.conv17 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn17   = nn.BatchNorm2d(32)
        self.conv18 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)

        #9
        self.bn18   = nn.BatchNorm2d(32)
        self.conv19 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn19   = nn.BatchNorm2d(32)
        self.conv20 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)

        #10
        self.bn20   = nn.BatchNorm2d(32)
        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn21   = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)

        #11 (Stage 3)
        self.bn22   = nn.BatchNorm2d(32)
        self.conv23 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn23   = nn.BatchNorm2d(64)
        self.conv24 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn24   = nn.BatchNorm2d(64)
        self.conv25 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False, stride=2)

        #12
        self.bn25   = nn.BatchNorm2d(64)
        self.conv26 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn26   = nn.BatchNorm2d(64)
        self.conv27 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)

        #13
        self.bn27   = nn.BatchNorm2d(64)
        self.conv28 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn28   = nn.BatchNorm2d(64)
        self.conv29 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)

        #14
        self.bn29   = nn.BatchNorm2d(64)
        self.conv30 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn30   = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)

        #15
        self.bn31   = nn.BatchNorm2d(64)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn32   = nn.BatchNorm2d(64)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)

        self.bn33   = nn.BatchNorm2d(64)
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
        x = self.bn1(_x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
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

        #4
        x = self.bn7(_x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        _x = _x + x

        #5
        x = self.bn9(_x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        _x = _x + x

        #6 (Stage 2)
        _x = self.bn11(_x)
        _x = self.relu(_x)
        x = self.conv12(_x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        _x = self.conv14(_x)
        _x = _x + x

        #7
        x = self.bn14(_x)
        x = self.relu(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu(x)
        x = self.conv16(x)
        _x = _x + x

        #8
        x = self.bn16(_x)
        x = self.relu(x)
        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu(x)
        x = self.conv18(x)
        _x = _x + x

        #9
        x = self.bn18(_x)
        x = self.relu(x)
        x = self.conv19(x)
        x = self.bn19(x)
        x = self.relu(x)
        x = self.conv20(x)
        _x = _x + x

        #10
        x = self.bn20(_x)
        x = self.relu(x)
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu(x)
        x = self.conv22(x)
        _x = _x + x

        #11 (Stage 3)
        _x = self.bn22(_x)
        _x = self.relu(_x)
        x = self.conv23(_x)
        x = self.bn23(x)
        x = self.relu(x)
        x = self.conv24(x)
        x = self.bn24(x)
        _x = self.conv25(_x)
        _x = _x + x

        #12
        x = self.bn25(_x)
        x = self.relu(x)
        x = self.conv26(x)
        x = self.bn26(x)
        x = self.relu(x)
        x = self.conv27(x)
        _x = _x + x

        #13
        x = self.bn27(_x)
        x = self.relu(x)
        x = self.conv28(x)
        x = self.bn28(x)
        x = self.relu(x)
        x = self.conv29(x)
        _x = _x + x

        #14
        x = self.bn29(_x)
        x = self.relu(x)
        x = self.conv30(x)
        x = self.bn30(x)
        x = self.relu(x)
        x = self.conv31(x)
        _x = _x + x

        #15
        x = self.bn31(_x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv33(x)
        _x = _x + x

        x = self.bn33(_x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet32_v2_flat(**kwargs):
    model = ResNet32(**kwargs)
    return model
