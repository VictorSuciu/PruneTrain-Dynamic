"""  All unique layers of AlexNet for CIFAR10/100
"""

import os
from .arch_utils import layerUtil

arch = {}
arch[0] = {'name':'conv1', 'kernel_size':11, 'stride':4, 'padding':5, 'bias':True}
arch[1] = {'name':'conv2', 'kernel_size':5,  'stride':1, 'padding':2, 'bias':True}
arch[2] = {'name':'conv3', 'kernel_size':3,  'stride':1, 'padding':1, 'bias':True}
arch[3] = {'name':'conv4', 'kernel_size':3,  'stride':1, 'padding':1, 'bias':True}
arch[4] = {'name':'conv5', 'kernel_size':3,  'stride':1, 'padding':1, 'bias':True}
arch[5] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[6] = {'name':'relu'}
arch[7] = {'name':'fc', 'out_chs':'num_classes'}


"""
Generate dense AelxNet architecture
- Only input/output channel number change
"""
def _genDenseArchAlexNet(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map=None):
  lyr_util = layerUtil(model)

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'alexnet_flat\']\n'
  ctx += 'class AlexNet(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(AlexNet, self).__init__()\n'

  # Layer definition
  for idx in sorted(arch):
    name = arch[idx]['name']
    if   'conv' in name:  ctx += lyr_util.convLayer(name, arch[idx])
    elif 'relu' == name:  ctx += lyr_util.reluLayer()
    elif 'pool' == name:  ctx += lyr_util.poolLayer(arch[idx])
    elif 'fc'   == name:  ctx += lyr_util.fcLayer(name, arch[idx])
    else: assert True, 'wrong layer name'

  # Architecture sequential
  def forward(name):
    return '\t\tx = self.{}(x)\n'.format(name)

  ctx += '\tdef forward(self, x):\n'
  ctx += forward('conv1')
  ctx += forward('relu')
  ctx += forward('pool')
  ctx += forward('conv2')
  ctx += forward('relu')
  ctx += forward('pool')
  ctx += forward('conv3')
  ctx += forward('relu')
  ctx += forward('conv4')
  ctx += forward('relu')
  ctx += forward('conv5')
  ctx += forward('relu')
  ctx += forward('pool')
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += forward('fc')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def alexnet_flat(**kwargs):\n'
  ctx += '\tmodel = AlexNet(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'alexnet_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)



