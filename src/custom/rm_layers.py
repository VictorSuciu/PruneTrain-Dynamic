# ResNet for CIFAR10/100
resnet20 = []
for i in [2,4,6,8,11,13,15,18,20]:
    resnet20.append(['module.conv'+str(i), 'module.conv'+str(i+1)])

resnet32 = []
for i in [2,4,6,8,10,12,15,17,19,21,23,26,28,30,32]:
    resnet32.append(['module.conv'+str(i), 'module.conv'+str(i+1)])

resnet50_bt = []
for i in [2,6,9,12,15,18,21,24,27,31,34,37,40,43,46,49,52,56,59,62,65,68,71,74]:
    resnet50_bt.append(['module.conv'+str(i), 'module.conv'+str(i+1), 'module.conv'+str(i+2)])

rm_pairs_cifar = {
    'resnet20_flat':resnet20, 
    'resnet32_flat':resnet32, 
    'resnet50_bt_flat':resnet50_bt, 
}

# ResNet for IMAGENET
resnet34 = []
for i in [2,4,6,8,11,13,15,17,20,22,24,26,28,30,33,35]:
    resnet34.append(['module.conv'+str(i), 'module.conv'+str(i+1)])

resnet50 = []
for i in [2,6,9,12,16,19,22,25,29,32,35,38,41,44,48,51]:
    resnet50.append(['module.conv'+str(i), 'module.conv'+str(i+1), 'module.conv'+str(i+2)])

rm_pairs_imgnet = {
    'resnet34_flat':resnet34, 
    'resnet50_flat':resnet50, 
}


""" Convolution layers in the same residual path
# When a convolution layer is removed in a residual path, all layers in the path are removed
"""
def getRmLayers(name, arch, dataset):
    name = name.split('.weight')[0]
    rm_layers = None

    if 'cifar' in dataset:
        layer_pairs = rm_pairs_cifar[arch]
    else:
        layer_pairs = rm_pairs_imgnet[arch]

    for layer_pair in layer_pairs:
        if name in layer_pair:
            rm_layers = layer_pair
            break

    if rm_layers != None:
        # Add BN layers to remove
        rm_bns = []
        for conv_name in rm_layers:
            bn_name = conv_name.replace('conv', 'bn')
            rm_bns.append(bn_name)
        rm_layers.extend(rm_bns)

        return rm_layers
    else:
        return []
