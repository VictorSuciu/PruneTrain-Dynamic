'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from custom.checkpoint_utils_fp32 import _makeSparse
from custom.checkpoint_utils_fp32 import _genDenseModel
from custom_arch import *
import numpy as np
import math as mt

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('--data_train', default='path to dataset', type=str)
parser.add_argument('--data_val', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--schedule-exp', type=int, default=0, help='Exponential LR decay.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#======= Custom variables. begin
parser.add_argument('--save_checkpoint', default=10, type=int, 
                    help='Interval to save checkpoint')
parser.add_argument('--sparse_interval', default=0, type=int, 
                    help='Interval to force the value under threshold')
parser.add_argument('--threshold', default=0.0001, type=float, 
                    help='Threshold to force weight to zero')
parser.add_argument('--en_group_lasso', default=False, action='store_true',
                    help='Set the group-lasso coefficient')
parser.add_argument('--var_group_lasso_coeff', default=0.1, type=float,
                    help='Ratio = group-lasso / (group-lasso + loss)')
parser.add_argument('--grp_lasso_coeff', default=0.0005, type=float,
                    help='claim as a global param')
parser.add_argument('--arch_out_dir', default=None, type=str,
                    help='directory to store the temporary architecture file')
parser.add_argument('--arch_name', default='net.py', type=str,
                    help='name of the new architecture')
parser.add_argument('--is_gating', default=False, action='store_true',
                    help='Use gating for residual network')
parser.add_argument('--threshold_type', default='max', choices=['max', 'mean'], type=str,
                    help='Thresholding type')
parser.add_argument('--coeff_container', default='./coeff', type=str,
                    help='Directory to store lasso coefficient')
parser.add_argument('--add_layer_lasso', default=False, action='store_true',
                    help='Apply layer-wise group lasso')
parser.add_argument('--l2_lasso_penalty', default=1., type=float, 
                    help='Extra penalty to overlapping lasso for layer removal')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)') 
parser.add_argument('--groups', type=int, default=16,
                    help='Number of groups or channels per group for group normalization.')
#======= Custom variables. end

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


# Sub-sampling dataset
class LimitDataset(data.Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


def main():

    # ==== TACC: copy ImageNet to local ====
#    if not os.path.exists('/tmp/dataset/imagenet-data'):
#        os.makedirs('/tmp/dataset/imagenet-data')
#        os.system('cp -r /work/03883/erhoo/projects/dataset/imagenet-data/raw-data /tmp/dataset/imagenet-data')

    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    traindir = args.data_train
    #valdir = os.path.join(args.data, 'validation')
    valdir = args.data_val
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,]))
    
    # Restrict the number of samples per class
    #train_dataset = LimitDataset(train_dataset, 300)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True)

    ################
    # create model
    ################
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnet50_gn'):
        if args.arch.startswith('resnet50_gn_mbs'):
            model = models.__dict__[args.arch](
                    max_split_sizes=args.max_split_sizes,
                    group_norm=args.groups)
        else:
            model = models.__dict__[args.arch](
                    group_norm=args.groups)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    if args.arch.startswith('alexnet'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    #for name, param in model.named_parameters():
    #    print("{}, {}".format(name, list(param.shape)))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Half precision setting
    #optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] +1 
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Lasso/Full_loss', 'Train Epoch Time(s)', 'Test Epoch Time(s)'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, test_epoch_time = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs+1):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

        train_loss, train_acc, lasso_ratio, train_epoch_time = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_epoch_time = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, lasso_ratio, train_epoch_time, test_epoch_time])


        # SparseTrain routine
        if args.en_group_lasso and (epoch % args.sparse_interval == 0):
            # Force weights under threshold to zero
            dense_chs, chs_map = _makeSparse(model, args.threshold, args.arch, 
                    args.threshold_type,
                    'imagenet',
                    is_gating=args.is_gating)
            # Reconstruct architecture
            if args.arch_out_dir != None:
                _genDenseModel(model, dense_chs, optimizer, args.arch, 'imagenet')
                _genDenseArch = custom_arch_imgnet[args.arch]
                if 'resnet' in args.arch:
                    _genDenseArch(model, args.arch_out_dir, args.arch_name, dense_chs, 
                            chs_map, args.is_gating)
                else:
                    _genDenseArch(model, args.arch_out_dir, args.arch_name, 
                            dense_chs, chs_map)

        # Save the checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print("[INFO] Storing checkpoint...")
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),},
                is_best, 
                checkpoint=args.checkpoint)

        if epoch % args.save_checkpoint == 0:
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),},
                    is_best, 
                    checkpoint=args.checkpoint,
                    filename='checkpoint'+str(epoch)+'.tar')

    logger.close()
    #logger.plot()
    #savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def _get_group_lasso(model):
    lasso_in_ch = []
    lasso_out_ch = []
    lasso_in_ch_penalty = []
    lasso_out_ch_penalty = []

    for name, param in model.named_parameters():
        # Lasso added to only the neuronal layers
        if ('weight' in name) and any([i for i in ['conv', 'fc'] if i in name]):
            if param.dim() == 4:
                conv_dw = int(name.split('.')[1].split('conv')[1]) %2 == 0
                #add_in_lasso = ('mobilenet' not in args.arch) or ('mobilenet' in args.arch and not conv_dw)
                add_lasso = ('mobilenet' not in args.arch) or ('mobilenet' in args.arch and not conv_dw)

                w_num_i_ch = param.shape[0] * param.shape[2] * param.shape[3]
                w_num_o_ch = param.shape[1] * param.shape[2] * param.shape[3]

                # Exclude depth-wise convolution layers from regularization
                if add_lasso:
                    if 'conv1.' not in name:
                        _in = param.pow(2).sum(dim=[0,2,3])
                        lasso_in_ch.append( _in )
                        penalty_tensor = torch.Tensor(param.shape[1]).cuda()
                        lasso_in_ch_penalty.append( penalty_tensor.new_full([param.shape[1]], w_num_i_ch) )

                    _out = param.pow(2).sum(dim=[1,2,3])
                    lasso_out_ch.append( _out )
                    penalty_tensor = torch.Tensor(param.shape[0]).cuda()
                    lasso_out_ch_penalty.append( penalty_tensor.new_full([param.shape[0]], w_num_o_ch) )

            elif param.dim() == 2:
                w_num_i_ch = param.shape[0]
                w_num_o_ch = param.shape[1]

                if ('fc1' in name) or ('fc2' in name):
                    lasso_out_ch.append( param.pow(2).sum(dim=[1]) )
                    penalty_tensor = torch.Tensor(param.shape[0]).cuda()
                    lasso_out_ch_penalty.append( penalty_tensor.new_full([param.shape[0]], w_num_o_ch) )
                lasso_in_ch.append( param.pow(2).sum(dim=[0]) )
                penalty_tensor = torch.Tensor(param.shape[1]).cuda()
                lasso_in_ch_penalty.append( penalty_tensor.new_full([param.shape[1]], w_num_i_ch) )

    _lasso_in_ch         = torch.cat(lasso_in_ch).cuda()
    _lasso_out_ch        = torch.cat(lasso_out_ch).cuda()

    lasso_penalty_in_ch  = _lasso_in_ch.add(1.0e-8).sqrt()
    lasso_penalty_out_ch = _lasso_out_ch.add(1.0e-8).sqrt()

    # Extra penalty using the number of parameters in each group
    lasso_in_ch_penalty  = torch.cat(lasso_in_ch_penalty).cuda().sqrt()
    lasso_out_ch_penalty  = torch.cat(lasso_out_ch_penalty).cuda().sqrt()

    lasso_penalty_in_ch  = lasso_penalty_in_ch.mul(lasso_in_ch_penalty).sum()
    lasso_penalty_out_ch = lasso_penalty_out_ch.mul(lasso_out_ch_penalty).sum()

    lasso_penalty        = lasso_penalty_in_ch + lasso_penalty_out_ch
    return lasso_penalty


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lasso_ratio = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        # sk. change back to float tensor
        loss = criterion(outputs, targets)

        # lasso penalty
        init_batch = batch_idx == 0 and epoch == 1
        lasso_time_start = time.time()

        if args.en_group_lasso:
            lasso_penalty = _get_group_lasso(model)

            # Auto-tune the group-lasso coefficient @first training iteration
            coeff_dir = os.path.join(args.coeff_container, 'imagenet', args.arch)
            if init_batch:
                args.grp_lasso_coeff = args.var_group_lasso_coeff *loss.item() / (lasso_penalty * (1-args.var_group_lasso_coeff))
                grp_lasso_coeff = torch.autograd.Variable(args.grp_lasso_coeff)

                if not os.path.exists( coeff_dir ):
                    os.makedirs( coeff_dir )
                with open( os.path.join(coeff_dir, str(args.var_group_lasso_coeff)), 'w' ) as f_coeff:
                    f_coeff.write( str(grp_lasso_coeff.item()) )

            else:
                with open( os.path.join(coeff_dir, str(args.var_group_lasso_coeff)), 'r' ) as f_coeff:
                    for line in f_coeff:
                        grp_lasso_coeff = float(line)

            lasso_penalty = lasso_penalty * grp_lasso_coeff
        else:
            lasso_penalty = 0.

        lasso_time = time.time() - lasso_time_start
        loss += lasso_penalty

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        lasso_ratio.update(lasso_penalty / loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        #batch_time.update(time.time() - end)
        batch_time.update(time.time() - end - lasso_time - data_load_time)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    epoch_time = batch_time.avg * len(train_loader)    # Time for total training dataset
    return (losses.avg, top1.avg, lasso_ratio.avg, epoch_time)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        #batch_time.update(time.time() - end)
        batch_time.update(time.time() - end - data_load_time)
        end = time.time()

    epoch_time = batch_time.avg * len(val_loader)   # Time for total test dataset
    return (losses.avg, top1.avg, epoch_time)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state

    if args.schedule_exp == 0:
        # Step-wise LR decay
        set_lr = args.lr
        for lr_decay in args.schedule:
            if epoch >= lr_decay:
                set_lr *= args.gamma
        state['lr'] = set_lr
    else:
        # Exponential LR decay
        set_lr = args.lr
        exp = int((epoch -1) / args.schedule_exp)
        state['lr'] = set_lr * (args.gamma**exp)

    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
