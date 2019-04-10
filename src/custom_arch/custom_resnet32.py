import os
from .arch_utils import layerUtil

""" 
All unique layers of ResNet32 for CIFAR10/100
"""
k3_s2_p1 = [12, 23]
k1_s2_p0 = [14, 25]

arch = {}
for i in range(1, 34):
  conv_idx = (i-1)*2
  bn_idx = conv_idx +1

  if i in k3_s2_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
  elif i in k1_s2_p0:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
  else:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
  arch[bn_idx] = {'name':'bn'+str(i)}

arch[66] = {'name':'avgpool', 'num':8}
arch[67] = {'name':'relu'}
arch[68] = {'name':'fc', 'out_chs':'num_classes'}


#arch = {}
#arch[0] = {'name':'conv1', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[1] = {'name':'bn1'}
#
#arch[2] = {'name':'conv2', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[3] = {'name':'bn2'}
#arch[4] = {'name':'conv3', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[5] = {'name':'bn3'}
#arch[6] = {'name':'conv4', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[7] = {'name':'bn4'}
#arch[8] = {'name':'conv5', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[9] = {'name':'bn5'}
#arch[10] = {'name':'conv6', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[11] = {'name':'bn6'}
#arch[12] = {'name':'conv7', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[13] = {'name':'bn7'}
#arch[14] = {'name':'conv8', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[15] = {'name':'bn8'}
#arch[16] = {'name':'conv9', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[17] = {'name':'bn9'}
#arch[18] = {'name':'conv10', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[19] = {'name':'bn10'}
#arch[20] = {'name':'conv11', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[21] = {'name':'bn11'}
#
#arch[22] = {'name':'conv12', 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
#arch[23] = {'name':'bn12'}
#arch[24] = {'name':'conv13', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[25] = {'name':'bn13'}
#arch[26] = {'name':'conv14', 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
#arch[27] = {'name':'bn14'}
#arch[28] = {'name':'conv15', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[29] = {'name':'bn15'}
#arch[30] = {'name':'conv16', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[31] = {'name':'bn16'}
#arch[32] = {'name':'conv17', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[33] = {'name':'bn17'}
#arch[34] = {'name':'conv18', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[35] = {'name':'bn18'}
#arch[36] = {'name':'conv19', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[37] = {'name':'bn19'}
#arch[38] = {'name':'conv20', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[39] = {'name':'bn20'}
#arch[40] = {'name':'conv21', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[41] = {'name':'bn21'}
#arch[42] = {'name':'conv22', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[43] = {'name':'bn22'}
#
#arch[44] = {'name':'conv23', 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
#arch[45] = {'name':'bn23'}
#arch[46] = {'name':'conv24', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[47] = {'name':'bn24'}
#arch[48] = {'name':'conv25', 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
#arch[49] = {'name':'bn25'}
#arch[50] = {'name':'conv26', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[51] = {'name':'bn26'}
#arch[52] = {'name':'conv27', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[53] = {'name':'bn27'}
#arch[54] = {'name':'conv28', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[55] = {'name':'bn28'}
#arch[56] = {'name':'conv29', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[57] = {'name':'bn29'}
#arch[58] = {'name':'conv30', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[59] = {'name':'bn30'}
#arch[60] = {'name':'conv31', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[61] = {'name':'bn31'}
#arch[62] = {'name':'conv32', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[63] = {'name':'bn32'}
#arch[64] = {'name':'conv33', 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
#arch[65] = {'name':'bn33'}

#arch[66] = {'name':'avgpool', 'num':8}
#arch[67] = {'name':'relu'}
#arch[68] = {'name':'fc', 'out_chs':'num_classes'}

"""
Generate dense ResNet32 architecture
- Only input/output channel number change
"""
def _genDenseArchResNet32(model, out_f_dir, arch_name, dense_chs, chs_map, is_gating=False):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += 'import torch\n'
  ctx += '__all__ = [\'resnet32_flat\']\n'
  ctx += 'class ResNet32(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(ResNet32, self).__init__()\n'

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
  ctx += lyr.resnet_module(chs_map0, is_gating, 8,9) #4
  ctx += lyr.resnet_module(chs_map0, is_gating, 10,11) #5

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 12,13,14) #6
  ctx += lyr.resnet_module(chs_map1, is_gating, 15,16) #7
  ctx += lyr.resnet_module(chs_map1, is_gating, 17,18) #8
  ctx += lyr.resnet_module(chs_map1, is_gating, 19,20) #9
  ctx += lyr.resnet_module(chs_map1, is_gating, 21,22) #10

  ctx += lyr.resnet_module_pool(chs_map1, chs_map2, is_gating, 23,24,25) #11
  ctx += lyr.resnet_module(chs_map2, is_gating, 26,27) #12
  ctx += lyr.resnet_module(chs_map2, is_gating, 28,29) #13
  ctx += lyr.resnet_module(chs_map2, is_gating, 30,31) #14
  ctx += lyr.resnet_module(chs_map2, is_gating, 32,33) #15

  if is_gating:
    ctx += lyr.mask('fc', chs_map2, i='_x', o='_x')

  ctx += '\t\tx = self.avgpool(_x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet32 definition
  ctx += 'def resnet32_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet32(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join('/workspace/models/pytorch-classification/models/cifar', 'resnet32_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir, arch_name),'w')
  f_out2.write(ctx)
