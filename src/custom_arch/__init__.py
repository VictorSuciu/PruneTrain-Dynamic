from .arch_utils import *

# CIFAR10/100
from .custom_alexnet import _genDenseArchAlexNet
#from .custom_vgg16_bn import _genDenseArchVGG16BN
from .custom_vgg13_bn import _genDenseArchVGG13BN
from .custom_vgg11_bn import _genDenseArchVGG11BN
from .custom_vgg8_bn import _genDenseArchVGG8BN
from .custom_resnet20 import _genDenseArchResNet20
from .custom_resnet32 import _genDenseArchResNet32
from .custom_resnet32_bt import _genDenseArchResNet32BT
from .custom_resnet50_bt import _genDenseArchResNet50BT

# IMAGENET
from .custom_resnet50 import _genDenseArchResNet50
from .custom_resnet50_01 import _genDenseArchResNet50_01
from .custom_resnet50_02 import _genDenseArchResNet50_02
from .custom_resnet50_03 import _genDenseArchResNet50_03
from .custom_resnet50_04 import _genDenseArchResNet50_04

from .custom_mobilenet import _genDenseArchMobileNet
from .custom_mobilenet_01 import _genDenseArchMobileNet_01
from .custom_mobilenet_02 import _genDenseArchMobileNet_02
from .custom_mobilenet_03 import _genDenseArchMobileNet_03

from .custom_vgg16_bn import _genDenseArchVGG16
from .custom_vgg16_bn_01 import _genDenseArchVGG16_01
from .custom_vgg16_bn_02 import _genDenseArchVGG16_02

custom_arch_cifar = {
    #'vgg16_bn_flat':_genDenseArchVGG16BN,
    'vgg13_bn_flat':_genDenseArchVGG13BN,
    'vgg11_bn_flat':_genDenseArchVGG11BN,
    'vgg8_bn_flat':_genDenseArchVGG8BN,
    'alexnet_flat':_genDenseArchAlexNet,
    'resnet20_flat':_genDenseArchResNet20,
    'resnet32_flat':_genDenseArchResNet32,
    'resnet32_bt_flat':_genDenseArchResNet32BT,
    'resnet50_bt_flat':_genDenseArchResNet50BT
}

custom_arch_imgnet = {
    'resnet50_flat':_genDenseArchResNet50,
    'resnet50_flat_01':_genDenseArchResNet50_01,
    'resnet50_flat_02':_genDenseArchResNet50_02,
    'resnet50_flat_03':_genDenseArchResNet50_03,
    'resnet50_flat_04':_genDenseArchResNet50_04,

    'mobilenet':_genDenseArchMobileNet,
    'mobilenet_01':_genDenseArchMobileNet_01,
    'mobilenet_02':_genDenseArchMobileNet_02,
    'mobilenet_03':_genDenseArchMobileNet_03,

    'vgg16_flat':_genDenseArchVGG16,
    'vgg16_flat_01':_genDenseArchVGG16_01,
    'vgg16_flat_02':_genDenseArchVGG16_02,
}
