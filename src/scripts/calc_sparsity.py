#!/home/sklym/anaconda2/bin/python

import os, sys
from os import listdir
from os.path import isfile, join
from statistics import mean

from ..src.custom.checkpoint_utils_fp32 import Checkpoint
from custom.visualize_utils import plotFilterSparsity, plotLayerSparsity, plotFilterData

MB = 1024*1024

#out_dir = '/home/sklym/Documents/sparse_train/cifar100_resnet32/sparse_train/auto_coef/0.1'
out_dir = './temp'
model_dir = '/work/03883/erhoo/projects/spar/sparse_train_pytorch/output/imagenet/resnet50/archive2/0.25'

check_point_names = [f for f in listdir(model_dir) if isfile(join(model_dir, f))  and 'checkpoint' in f]
#temp = {}
#for check_point_name in check_point_names:
#    temp[int(check_point_name.split('.')[1])] = check_point_name
#check_point_names = [f[1] for f in sorted(temp.items())]

#temp = []
#for check_point_name in check_point_names:
#    if "checkpoint.240.tar" in check_point_name:
#        temp.append(check_point_name)
temp = ['checkpoint90.tar']

check_point_names = temp

arch = "resnet50_flat"
threshold = 0.0001
depth = 20
num_classes = 1000

gen_figs = True
get_fil_data = True
target_lyr = ['conv40', 'conv41', 'conv42']
#target_lyr = ['conv8']

def calcConvSparsity(epochs, out_dir):
    avg_in_by_epoch  ={}
    avg_out_by_epoch  ={}
    max_epoch = 0

    for e in epochs:
        lyrs_density = epochs[e]

        list_in = []
        list_out = []

        for l in lyrs_density:
            list_in.append(lyrs_density[l]['in_ch'])
            list_out.append(lyrs_density[l]['out_ch'])

        avg_in_by_epoch[e] = mean(list_in)
        avg_out_by_epoch[e] = mean(list_out)
        max_epoch = max(max_epoch, e)

    print("========= input channel density ==========")
    for e in epochs:
        print ("{}, {}".format(e, str(avg_in_by_epoch[e])))

    print("========= output channel density ==========")
    for e in epochs:
        print ("{}, {}".format(e, str(avg_out_by_epoch[e])))

def main():
    conv_density_epochs = {}
    sparse_val_maps = {}

    for idx, check_point_name in enumerate(check_point_names):
        print ("Processing check_point: " +os.path.join(model_dir, check_point_name))
        model = Checkpoint(arch, 
                           os.path.join(model_dir, check_point_name), 
                           num_classes)

        if idx == 0 : model.printParams()

        # Generate conv layer sparsity
        sparse_bi_map, sparse_val_map, num_lyrs, conv_density, model_size, inf_cost =\
                model.getConvStructSparsity(threshold, out_dir+"/out_txt")
        
        if get_fil_data:
            fil_data = model.getFilterData(target_lyr)

        sparse_val_maps[idx] = sparse_val_map                
        print ("==> Model_size: {}, inference_cost: {}".format(model_size / MB, inf_cost))
        conv_density_epochs[model.getEpoch()] = conv_density

        #if gen_figs:
        #    plotFilterSparsity(check_point_name, sparse_bi_map, threshold, out_dir, num_lyrs)

    calcConvSparsity(conv_density_epochs, out_dir)

    #if gen_figs:
    #    plotLayerSparsity(sparse_val_maps, out_dir)
    #    if get_fil_data:
    #        plotFilterData(fil_data, out_dir)

if __name__ == "__main__":
    main()
