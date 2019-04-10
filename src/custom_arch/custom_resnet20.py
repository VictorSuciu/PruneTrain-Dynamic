import os
from .arch_utils import layerUtil

""" 
All unique layers of ResNet20 for CIFAR10/100
"""
arch = {}
arch[0] = {'name':'conv1', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[1] = {'name':'bn1'}

arch[2] = {'name':'conv2', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[3] = {'name':'bn2'}
arch[4] = {'name':'conv3', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[5] = {'name':'bn3'}
arch[6] = {'name':'conv4', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[7] = {'name':'bn4'}
arch[8] = {'name':'conv5', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[9] = {'name':'bn5'}
arch[10] = {'name':'conv6', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[11] = {'name':'bn6'}
arch[12] = {'name':'conv7', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[13] = {'name':'bn7'}

arch[14] = {'name':'conv8', 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
arch[15] = {'name':'bn8'}
arch[16] = {'name':'conv9', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[17] = {'name':'bn9'}
arch[18] = {'name':'conv10', 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
arch[19] = {'name':'bn10'}
arch[20] = {'name':'conv11', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[21] = {'name':'bn11'}
arch[22] = {'name':'conv12', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[23] = {'name':'bn12'}
arch[24] = {'name':'conv13', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[25] = {'name':'bn13'}
arch[26] = {'name':'conv14', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[27] = {'name':'bn14'}

arch[28] = {'name':'conv15', 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
arch[29] = {'name':'bn15'}
arch[30] = {'name':'conv16', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[31] = {'name':'bn16'}
arch[32] = {'name':'conv17', 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
arch[33] = {'name':'bn17'}
arch[34] = {'name':'conv18', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[35] = {'name':'bn18'}
arch[36] = {'name':'conv19', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[37] = {'name':'bn19'}
arch[38] = {'name':'conv20', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[39] = {'name':'bn20'}
arch[40] = {'name':'conv21', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[41] = {'name':'bn21'}
arch[42] = {'name':'avgpool', 'num':8}
arch[43] = {'name':'relu'}
arch[44] = {'name':'fc', 'out_chs':'num_classes'}

"""
Generate dense ResNet20 architecture
- Only input/output channel number change
"""
def _genDenseArchResNet20(model, out_f_dir, arch_name, dense_chs, chs_map, is_gating=False):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += 'import torch\n'
  ctx += '__all__ = [\'resnet20_flat\']\n'
  ctx += 'class ResNet20(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(ResNet20, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu', o='_x')

  if chs_map != None: chs_map0, chs_map1, chs_map2 = chs_map[0], chs_map[1], chs_map[2]
  else:               chs_map0, chs_map1, chs_map2 = None, None, None

  if is_gating:
    ctx += lyr.empty_ch(i='_x')
    ctx += lyr.merge('conv1', chs_map0, i='_x', o='_x')

  ctx += lyr.resnet_module(chs_map0, is_gating, 2,3) #1
  ctx += lyr.resnet_module(chs_map0, is_gating, 4,5) #2
  ctx += lyr.resnet_module(chs_map0, is_gating, 6,7) #3

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 8,9,10) #4
  ctx += lyr.resnet_module(chs_map1, is_gating, 11,12) #5
  ctx += lyr.resnet_module(chs_map1, is_gating, 13,14) #6

  ctx += lyr.resnet_module_pool(chs_map1, chs_map2, is_gating, 15,16,17) #7
  ctx += lyr.resnet_module(chs_map2, is_gating, 18,19) #8
  ctx += lyr.resnet_module(chs_map2, is_gating, 20,21) #9

  if is_gating:
    ctx += lyr.mask('fc', chs_map2, i='_x', o='_x')

  ctx += '\t\tx = self.avgpool(_x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet20 definition
  ctx += 'def resnet20_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet20(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join('/workspace/models/pytorch-classification/models/cifar', 'resnet20_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir, 'resnet20_flat.py'),'w')
  f_out2.write(ctx)

