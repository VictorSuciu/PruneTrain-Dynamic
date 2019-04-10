import os
from .arch_utils import layerUtil

""" 
All unique layers of VGG8_BN for CIFAR10/100
"""
arch = {}
arch[0] = {'name':'conv1' , 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[1] = {'name':'bn1'}
arch[4] = {'name':'conv2' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[5] = {'name':'bn2'}
arch[8] = {'name':'conv3' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[9] = {'name':'bn3'}
arch[12] = {'name':'conv4' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[13] = {'name':'bn4'}
arch[16] = {'name':'conv5' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[17] = {'name':'bn5'}
arch[26] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[27] = {'name':'relu'}
arch[28] = {'name':'fc', 'out_chs':'num_classes'}

"""
Generate dense VGG8_BN architecture
- Only input/output channel number change
"""
def _genDenseArchVGG8BN(model, out_f_dir, dense_chs, chs_map=None):
  print ("[INFO] Generating a new dense architecture...")

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'vgg8_bn_flat\']\n'
  ctx += 'class VGG8(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(VGG8, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv2')
  ctx += lyr.forward('bn2')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv3')
  ctx += lyr.forward('bn3')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv4')
  ctx += lyr.forward('bn4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv5')
  ctx += lyr.forward('bn5')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def vgg8_bn_flat(**kwargs):\n'
  ctx += '\tmodel = VGG8(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  f_out = open(os.path.join(out_f_dir, 'vgg8_bn_flat.py'),'w')
  f_out.write(ctx)
