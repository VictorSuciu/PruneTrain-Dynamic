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
arch[42] = {'name':'relu'}
arch[43] = {'name':'fc', 'out_chs':'num_classes'}


"""
Generate dense AelxNet architecture
- Only input/output channel number change
"""
def _genDenseArchResNet20v2(model, out_f_dir, arch_name, dense_chs, chs_map):
  lyr = layerUtil(model)

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'resnet20_v2_flat\']\n'
  ctx += 'class ResNet20(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(ResNet20, self).__init__()\n'

  # Layer definition
  for idx in sorted(arch):
    name = arch[idx]['name']
    if   'conv' in name:  ctx += lyr.convLayer(name, arch[idx])
    elif 'relu' == name:  ctx += lyr.reluLayer()
    elif 'pool' == name:  ctx += lyr.poolLayer(arch[idx])
    elif 'fc'   == name:  ctx += lyr.fcLayer(name, arch[idx])
    else: assert True, 'wrong layer name'

  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1', o='_x')
  ctx += merge(dense_chs['conv1'], chs_map[0])

  #1
  ctx += mask(dense_chs['conv2'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn1', i='__x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv2')
  ctx += lyr.forward('bn2')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv3')
  ctx += merge(dense_chs['conv3'], chs_map[0])
  ctx += lyr.sum()

  #2
  ctx += mask(dense_chs['conv4'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn3', i='_x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv4')
  ctx += lyr.forward('bn4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv5')
  ctx += merge(dense_chs['conv5'], chs_map[0])
  ctx += lyr.sum()

  #3
  ctx += mask(dense_chs['conv6'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn5', i='_x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv6')
  ctx += lyr.forward('bn6')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv7')
  ctx += merge(dense_chs['conv7'], chs_map[0])
  ctx += lyr.sum()

  #4 (Stage 2)
  ctx += mask(dense_chs['conv8'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn7', i='_x', o='_x')
  ctx += lyr.forward('relu', i='_x', o='_x')
  ctx += lyr.forward('conv8', i='_x')
  ctx += lyr.forward('bn8')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv9')
  ctx += merge(dense_chs['conv9'], chs_map[1])
  ctx += lyr.forward('conv10', i='_x', o='_x')
  ctx += merge(dense_chs['conv10'], chs_map[1])
  ctx += lyr.sum()
  
  #5
  ctx += mask(dense_chs['conv11'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn10', i='_x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv11')
  ctx += lyr.forward('bn11')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv12')
  ctx += merge(dense_chs['conv12'], chs_map[1])
  ctx += lyr.sum()

  #6
  ctx += mask(dense_chs['conv13'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn12', i='_x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv13')
  ctx += lyr.forward('bn13')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv14')
  ctx += merge(dense_chs['conv14'], chs_map[1])
  ctx += lyr.sum()

  #7 (Stage 3)
  ctx += mask(dense_chs['conv15'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn14', i='_x', o='_x')
  ctx += lyr.forward('relu', i='_x', o='_x')
  ctx += lyr.forward('conv15', i='_x')
  ctx += lyr.forward('bn15')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv16')
  ctx += merge(dense_chs['conv16'], chs_map[2])
  ctx += lyr.forward('conv17', i='_x', o='_x')
  ctx += merge(dense_chs['conv17'], chs_map[2])
  ctx += lyr.sum()

  #8
  ctx += mask(dense_chs['conv18'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn17', i='_x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv18')
  ctx += lyr.forward('bn18')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv19')
  ctx += merge(dense_chs['conv19'], chs_map[2])
  ctx += lyr.sum()

  #9
  ctx += mask(dense_chs['conv20'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn19', i='_x')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv20')
  ctx += lyr.forward('bn20')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv21')
  ctx += merge(dense_chs['conv21'], chs_map[2])
  ctx += lyr.sum()

  ctx += mask(dense_chs['conv20'], chs_map[0], i='_x', o='__x')
  ctx += lyr.forward('bn21', i='_x')
  ctx += lyr.forward('relu')
  ctx += '\t\tx = self.avgpool(x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet20 definition
  ctx += 'def resnet20_v2_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet20(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join('/workspace/models/pytorch-classification/models/cifar', 'resnet20_v2_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir, arch_name),'w')
  f_out2.write(ctx)


