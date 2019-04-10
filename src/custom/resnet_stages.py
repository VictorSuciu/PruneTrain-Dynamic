def n(name):
    if isinstance(name, int):
        return 'module.conv'+str(name)+'.weight'
    else:
        return 'module.'+name+'.weight'

# =============================================== ResNet20
resnet20_cifar = {0:{}, 1:{}, 2:{}, 10:{}}
resnet20_cifar[0]['i'] = [n(2), n(4), n(6), n(8), n(10)]
resnet20_cifar[0]['o'] = [n(1), n(3), n(5), n(7)]
resnet20_cifar[1]['i'] = [n(11), n(13), n(15), n(17)]
resnet20_cifar[1]['o'] = [n(9), n(10), n(12), n(14)]
resnet20_cifar[2]['i'] = [n(18), n(20), n('fc')]
resnet20_cifar[2]['o'] = [n(16), n(17), n(19), n(21)]

# A sequential layer pairs within residual paths
resnet20_cifar[10] = [
    [n(2), n(3)], [n(4), n(5)], [n(6), n(7)], [n(8), n(9)],
    [n(11), n(12)], [n(13), n(14)], [n(15), n(16)], [n(18), n(19)],
    [n(20), n(21)]
]

# =============================================== ResNet32
resnet32_cifar = {0:{}, 1:{}, 2:{}, 10:{}}
resnet32_cifar[0]['i'] = [n(2), n(4), n(6), n(8), n(10), n(12), n(14)]
resnet32_cifar[0]['o'] = [n(1), n(3), n(5), n(7), n(9), n(11)]
resnet32_cifar[1]['i'] = [n(15), n(17), n(19), n(21), n(23), n(25)]
resnet32_cifar[1]['o'] = [n(13), n(14), n(16), n(18), n(20), n(22)]
resnet32_cifar[2]['i'] = [n(26), n(28), n(30), n(32), n('fc')]
resnet32_cifar[2]['o'] = [n(24), n(25), n(27), n(29), n(31), n(33)]

# A sequential layer pairs within residual paths
resnet32_cifar[10] = [
    [n(2), n(3)],   [n(4), n(5)],   [n(6), n(7)],   [n(8), n(9)],
    [n(10), n(11)], [n(12), n(13)], [n(15), n(16)], [n(17), n(18)],
    [n(19), n(20)], [n(21), n(22)], [n(23), n(24)], [n(26), n(27)],
    [n(28), n(29)], [n(30), n(31)], [n(32), n(33)]
]

# =============================================== ResNet32_BT
resnet32_bt_cifar = {0:{}, 1:{}, 2:{}, 3:{}, 10:{}}
resnet32_bt_cifar[0]['i'] = [n(2), n(5)]
resnet32_bt_cifar[0]['o'] = [n(1)]
resnet32_bt_cifar[1]['i'] = [n(6), n(9), n(12), n(15), n(18), n(21)]
resnet32_bt_cifar[1]['o'] = [n(4), n(5), n(8), n(11), n(14), n(17)]
resnet32_bt_cifar[2]['i'] = [n(22), n(25), n(28), n(31), n(34), n(37)]
resnet32_bt_cifar[2]['o'] = [n(20), n(21), n(24), n(27), n(30), n(33)]
resnet32_bt_cifar[3]['i'] = [n(38), n(41), n(44), n(47), n('fc')]
resnet32_bt_cifar[3]['o'] = [n(36), n(37), n(40), n(43), n(46), n(49)]

# A sequential layer pairs within residual paths
resnet32_bt_cifar[10] = [
    [n(2),  n(3),  n(4)],  [n(6),  n(7),  n(8)],  [n(9),  n(10), n(11)], [n(12), n(13), n(14)],
    [n(15), n(16), n(17)], [n(18), n(19), n(20)], [n(22), n(23), n(24)], [n(25), n(26), n(27)],
    [n(28), n(29), n(30)], [n(31), n(32), n(33)], [n(34), n(35), n(36)], [n(38), n(39), n(40)],
    [n(41), n(42), n(43)], [n(44), n(45), n(46)], [n(47), n(48), n(49)]
]

# =============================================== ResNet50_BT
resnet50_bt_cifar = {0:{}, 1:{}, 2:{}, 3:{}, 10:{}}
resnet50_bt_cifar[0]['i'] = [n(2), n(5)]
resnet50_bt_cifar[0]['o'] = [n(1)]
resnet50_bt_cifar[1]['i'] = [n(6), n(9), n(12), n(15), n(18), n(21), n(24), n(27), n(30)]
resnet50_bt_cifar[1]['o'] = [n(4), n(5), n(8), n(11), n(14), n(17), n(20), n(23), n(26)]
resnet50_bt_cifar[2]['i'] = [n(31), n(34), n(37), n(40), n(43), n(46), n(49), n(52), n(55)]
resnet50_bt_cifar[2]['o'] = [n(29), n(30), n(33), n(36), n(39), n(42), n(45), n(48), n(51)]
resnet50_bt_cifar[3]['i'] = [n(56), n(59), n(62), n(65), n(68), n(71), n(74), n('fc')]
resnet50_bt_cifar[3]['o'] = [n(54), n(55), n(58), n(61), n(64), n(67), n(70), n(73), n(76)]

# A sequential layer pairs within residual paths
resnet50_bt_cifar[10] = [
    [n(2),  n(3),  n(4)],  [n(6),  n(7),  n(8)],  [n(9),  n(10), n(11)], [n(12), n(13), n(14)],
    [n(15), n(16), n(17)], [n(18), n(19), n(20)], [n(21), n(22), n(23)], [n(24), n(25), n(26)],
    [n(27), n(28), n(29)], [n(31), n(32), n(33)], [n(34), n(35), n(36)], [n(37), n(38), n(39)],
    [n(40), n(41), n(42)], [n(43), n(44), n(45)], [n(46), n(47), n(48)], [n(49), n(50), n(51)],
    [n(52), n(53), n(54)], [n(56), n(57), n(58)], [n(59), n(60), n(61)], [n(62), n(63), n(64)],
    [n(65), n(66), n(67)], [n(68), n(69), n(70)], [n(71), n(72), n(73)], [n(74), n(75), n(76)]
]

stages_cifar = {
    'resnet20_v2_flat':resnet20_cifar,
    'resnet20_flat':resnet20_cifar,
    'resnet32_flat':resnet32_cifar,
    'resnet32_flat_temp':resnet32_cifar,
    'resnet32_bt_flat':resnet32_bt_cifar,
    'resnet50_bt_flat':resnet50_bt_cifar,
}

# =================================================================
# =============================== ImageNet
# =================================================================

# =============================================== ResNet50_BT
resnet50_imgnet = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 10:{}}
resnet50_imgnet[0]['i'] = [n(2), n(5)]
resnet50_imgnet[0]['o'] = [n(1)]
resnet50_imgnet[1]['i'] = [n(6),  n(9),  n(12), n(15)]
resnet50_imgnet[1]['o'] = [n(4),  n(5),  n(8),  n(11)]
resnet50_imgnet[2]['i'] = [n(16), n(19), n(22), n(25), n(28)]
resnet50_imgnet[2]['o'] = [n(14), n(15), n(18), n(21), n(24)]
resnet50_imgnet[3]['i'] = [n(29), n(32), n(35), n(38), n(41), n(44), n(47)]
resnet50_imgnet[3]['o'] = [n(27), n(28), n(31), n(34), n(37), n(40), n(43)]
resnet50_imgnet[4]['i'] = [n(48), n(51), n('fc')]
resnet50_imgnet[4]['o'] = [n(46), n(47), n(50), n(53)]

# A sequential layer pairs within residual paths
resnet50_imgnet[10] = [
    [n(2),  n(3),  n(4)],  [n(6),  n(7),  n(8)],  [n(9),  n(10), n(11)], [n(12), n(13), n(14)],
    [n(16), n(17), n(18)], [n(19), n(20), n(21)], [n(22), n(23), n(24)], [n(25), n(26), n(27)],
    [n(29), n(30), n(31)], [n(32), n(33), n(34)], [n(35), n(36), n(37)], [n(38), n(39), n(40)],
    [n(41), n(42), n(43)], [n(44), n(45), n(46)], [n(48), n(49), n(50)], [n(51), n(52), n(53)],
]

stages_imgnet = {
    'resnet50_flat':resnet50_imgnet,
    'resnet50_flat_01':resnet50_imgnet,
    'resnet50_flat_02':resnet50_imgnet,
    'resnet50_flat_03':resnet50_imgnet,
    'resnet50_flat_04':resnet50_imgnet,
    'resnet50_gn':resnet50_imgnet,
}




