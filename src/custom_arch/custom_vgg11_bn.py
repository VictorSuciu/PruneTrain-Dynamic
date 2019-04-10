import os
from .arch_utils import layerUtil

""" 
All unique layers of VGG11_BN for CIFAR10/100
"""
arch = {}
arch[0] = {'name':'conv1' , 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[1] = {'name':'bn1'}

arch[4] = {'name':'conv2' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[5] = {'name':'bn2'}

arch[8] = {'name':'conv3' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[9] = {'name':'bn3'}
arch[10] = {'name':'conv4' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[11] = {'name':'bn4'}

arch[12] = {'name':'conv5' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[13] = {'name':'bn5'}
arch[14] = {'name':'conv6' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[15] = {'name':'bn6'}

arch[16] = {'name':'conv7' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[17] = {'name':'bn7'}
arch[18] = {'name':'conv8',  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[19] = {'name':'bn8'}

arch[26] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[27] = {'name':'relu'}
arch[28] = {'name':'fc', 'out_chs':'num_classes'}

"""
Generate dense VGG11_BN architecture
- Only input/output channel number change
"""
def _genDenseArchVGG11BN(model, out_f_dir, arch_name, dense_chs, chs_map=None):
  print ("[INFO] Generating a new dense architecture...")

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'vgg11_bn_flat\']\n'
  ctx += 'class VGG11(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(VGG11, self).__init__()\n'

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
  ctx += lyr.forward('conv4')
  ctx += lyr.forward('bn4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv5')
  ctx += lyr.forward('bn5')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv6')
  ctx += lyr.forward('bn6')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv7')
  ctx += lyr.forward('bn7')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv8')
  ctx += lyr.forward('bn8')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def vgg11_bn_flat(**kwargs):\n'
  ctx += '\tmodel = VGG11(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  f_out1 = open(os.path.join('/workspace/models/pytorch-classification/models/cifar', 'vgg11_bn_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir, arch_name),'w')
  f_out2.write(ctx)

