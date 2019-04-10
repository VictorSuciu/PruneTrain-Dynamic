import os
from .arch_utils import layerUtil

""" 
All unique layers of ResNet32BT for CIFAR10/100
"""

k3_s2_p1 = [19, 35]
k1_s2_p0 = [21, 37]
k3_s1_p1 = [1, 3, 7, 10, 13, 16, 19,    23, 26, 29, 32, 35,    39, 42, 45, 48]

arch = {}
for i in range(1, 77):
  conv_idx = (i-1)*2
  bn_idx   = conv_idx +1

  if i in k3_s2_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
  elif i in k1_s2_p0:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
  elif i in k3_s1_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
  else:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':1, 'padding':0, 'bias':False}
  arch[bn_idx] = {'name':'bn'+str(i)}

arch[99] = {'name':'avgpool', 'num':8}
arch[100] = {'name':'relu'}
arch[101] = {'name':'fc', 'out_chs':'num_classes'}

"""
Generate dense ResNet32BT architecture
- Only input/output channel number change
"""
def _genDenseArchResNet32BT(model, out_f_dir, arch_name, dense_chs, chs_map, is_gating=False):
  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += 'import torch\n'
  ctx += '__all__ = [\'resnet32_bt_flat\']\n'
  ctx += 'class ResNet32BT(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(ResNet32BT, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu', o='_x')

  if chs_map != None: chs_map0, chs_map1, chs_map2, chs_map3 = chs_map[0], chs_map[1], chs_map[2], chs_map[3]
  else:               chs_map0, chs_map1, chs_map2, chs_map3 = None, None, None, None

  if is_gating:
    ctx += lyr.empty_ch(i='_x')
    ctx += lyr.merge('conv1', chs_map0, i='_x', o='_x')

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 2,3,4,5) #1
  ctx += lyr.resnet_module(chs_map1, is_gating, 6,7,8) #2
  ctx += lyr.resnet_module(chs_map1, is_gating, 9,10,11) #3
  ctx += lyr.resnet_module(chs_map1, is_gating, 12,13,14) #4
  ctx += lyr.resnet_module(chs_map1, is_gating, 15,16,17) #5

  ctx += lyr.resnet_module_pool(chs_map1, chs_map2, is_gating, 18,19,20,21) #6
  ctx += lyr.resnet_module(chs_map2, is_gating, 22,23,24) #7
  ctx += lyr.resnet_module(chs_map2, is_gating, 25,26,27) #8
  ctx += lyr.resnet_module(chs_map2, is_gating, 28,29,30) #9
  ctx += lyr.resnet_module(chs_map2, is_gating, 31,32,33) #10

  ctx += lyr.resnet_module_pool(chs_map2, chs_map3, is_gating, 34,35,36,37) #11
  ctx += lyr.resnet_module(chs_map3, is_gating, 38,39,40) #12
  ctx += lyr.resnet_module(chs_map3, is_gating, 41,42,43) #13
  ctx += lyr.resnet_module(chs_map3, is_gating, 44,45,46) #14
  ctx += lyr.resnet_module(chs_map3, is_gating, 47,48,49) #15

  if is_gating:
    ctx += lyr.mask('fc', chs_map3, i='_x', o='_x')

  ctx += '\t\tx = self.avgpool(_x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet32BT definition
  ctx += 'def resnet32_bt_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet32BT(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join('/workspace/models/pytorch-classification/models/cifar', 'resnet32_bt_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir, arch_name),'w')
  f_out2.write(ctx)
