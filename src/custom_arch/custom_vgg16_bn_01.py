import os
from .arch_utils import layerUtil

""" 
All unique layers of VGG16_BN for CIFAR10/100
"""
arch = {}
arch[0] = {'name':'conv1' , 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[1] = {'name':'bn1'}
arch[2] = {'name':'conv2' , 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[3] = {'name':'bn2'}
arch[4] = {'name':'conv3' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[5] = {'name':'bn3'}
arch[6] = {'name':'conv4' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[7] = {'name':'bn4'}
arch[8] = {'name':'conv5' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[9] = {'name':'bn5'}
arch[10] = {'name':'conv6' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[11] = {'name':'bn6'}
arch[12] = {'name':'conv7' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[13] = {'name':'bn7'}
arch[14] = {'name':'conv8' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[15] = {'name':'bn8'}
arch[16] = {'name':'conv9' ,  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[17] = {'name':'bn9'}
arch[18] = {'name':'conv10',  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[19] = {'name':'bn10'}
arch[20] = {'name':'conv11',  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[21] = {'name':'bn11'}
arch[22] = {'name':'conv12',  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[23] = {'name':'bn12'}
arch[24] = {'name':'conv13',  'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
arch[25] = {'name':'bn13'}
arch[26] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[27] = {'name':'avgpool_adt', 'num':'(7, 7)'}
arch[28] = {'name':'relu'}
arch[29] = {'name':'fc1'}
arch[30] = {'name':'fc2'}
arch[31] = {'name':'fc3'}


"""
Generate dense VGG16_BN architecture
- Only input/output channel number change
"""
def _genDenseArchVGG16_01(model, out_f_dir, arch_name, dense_chs, chs_map=None):
  print ("[INFO] Generating a new dense architecture...")

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'vgg16_flat_01\']\n'
  ctx += 'class VGG16(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=1000):\n'
  ctx += '\t\tsuper(VGG16, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu')
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
  ctx += lyr.forward('conv7')
  ctx += lyr.forward('bn7')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv8')
  ctx += lyr.forward('bn8')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv9')
  ctx += lyr.forward('bn9')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv10')
  ctx += lyr.forward('bn10')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv11')
  ctx += lyr.forward('bn11')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv12')
  ctx += lyr.forward('bn12')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv13')
  ctx += lyr.forward('bn13')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('avgpool')
  
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc1')
  ctx += lyr.forward('relu')
  ctx += '\t\tx = self.dropout(x)\n'
  ctx += lyr.forward('fc2')
  ctx += lyr.forward('relu')
  ctx += '\t\tx = self.dropout(x)\n'
  ctx += lyr.forward('fc3')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def vgg16_flat_01(**kwargs):\n'
  ctx += '\tmodel = VGG16(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir):
      os.makedirs(out_f_dir)

  f_out1 = open(os.path.join('/work/03883/erhoo/projects/spar/sparse_train_pytorch/src/models/imagenet', 'vgg16_flat_01.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir, arch_name),'w')
  f_out2.write(ctx)


