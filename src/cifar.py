'''
Training script for CIFAR-10/100
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
import models.cifar as models

#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

from custom.checkpoint_utils import _makeSparse
from custom.checkpoint_utils import _genDenseModel
from custom_arch import *
import numpy as np

from apex.fp16_utils import FP16_Optimizer

MB = 1024*1024

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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
parser.add_argument('--sparse_train', action='store_true',
                    help='Force the weights in the middle of training')
parser.add_argument('--en_auto_lasso_coeff', default=False, action='store_true',
                    help='Set the group-lasso coefficient')
parser.add_argument('--var_auto_lasso_coeff', default=0.1, type=float,
                    help='Ratio = group-lasso / (group-lasso + loss)')
parser.add_argument('--grp_lasso_coeff', default=0.0005, type=float,
                    help='claim as a global param')
parser.add_argument('--resnet_v2', default=True, action='store_true',
                    help='resnet_v2 flag')
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
#======= Custom variables. end

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

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

def main():
    global best_acc

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./dataset/data/torch', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, 
                                batch_size=args.train_batch, 
                                shuffle=True, 
                                num_workers=args.workers)

    testset = dataloader(root='./dataset/data/torch', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.endswith('resnet_v2'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())*4/MB))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Lasso/Full_loss', 'Train Epoch Time(s)', 'Test Epoch Time(s)'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, lasso_ratio, train_epoch_time = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_epoch_time = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, lasso_ratio, train_epoch_time, test_epoch_time])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if args.sparse_train and ((epoch +1) % args.sparse_interval == 0):
            # Force weights under threshold to zero
            dense_chs, chs_map = _makeSparse(model, args.threshold, args.arch, is_gating=args.is_gating)

            # Reconstruct architecture
            if args.arch_out_dir != None:
                _genDenseModel(model, dense_chs, optimizer, args.arch)
                _genDenseArch = custom_arch[args.arch]
                if 'resnet' in args.arch:
                    _genDenseArch(model, args.arch_out_dir, args.arch_name, dense_chs, chs_map, args.is_gating)
                else:
                    _genDenseArch(model, args.arch_out_dir, args.arch_name, dense_chs, chs_map)

        if (epoch +1) % args.save_checkpoint == 0:
            print("[INFO] Storing checkpoint...")
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),}, 
                    is_best, 
                    checkpoint=args.checkpoint, 
                    filename='checkpoint.'+str(epoch +1)+'.tar')

    logger.close()
    #logger.plot()
    #savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def _get_group_lasso(model):

    lasso_in_ch = []
    lasso_out_ch = []

    for name, param in model.named_parameters():
        # Lasso added to BN parameters too
        #if 'weight' in name:

        # Lasso added to only the neuronal layers
        if ('weight' in name) and any([i for i in ['conv', 'fc'] if i in name]):
            if param.dim() == 4:
                if 'conv1.' not in name:
                    lasso_in_ch.append( param.pow(2).sum(dim=[0,2,3]) )
                lasso_out_ch.append( param.pow(2).sum(dim=[1,2,3]) )
            elif param.dim() == 2:
                lasso_in_ch.append( param.pow(2).sum(dim=[0]) )

    _lasso_in_ch         = torch.cat(lasso_in_ch).cuda()
    _lasso_out_ch        = torch.cat(lasso_out_ch).cuda()
    lasso_penalty_in_ch  = _lasso_in_ch.add(1.0e-8).sqrt().sum()
    lasso_penalty_out_ch = _lasso_out_ch.add(1.0e-8).sqrt().sum()
    lasso_penalty        = lasso_penalty_in_ch + lasso_penalty_out_ch

    return lasso_penalty


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lasso_ratio = AverageMeter()

    end = time.time()

    #bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        #if epoch > 10:
        #    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #        outputs = model(inputs)
        #    print(prof)
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        # lasso penalty
        init_batch = batch_idx == 0 and epoch == 0

        lasso_time_start = time.time()

        if args.en_auto_lasso_coeff:
            lasso_penalty = _get_group_lasso(model)
            coeff_dir = os.path.join(args.coeff_container, args.dataset, args.arch)
            if init_batch:
                args.grp_lasso_coeff = args.var_auto_lasso_coeff *loss.item() / (lasso_penalty * (1-args.var_auto_lasso_coeff))
                grp_lasso_coeff = torch.autograd.Variable(args.grp_lasso_coeff)

                if not os.path.exists( coeff_dir ):
                    os.makedirs( coeff_dir )
                with open( os.path.join(coeff_dir, str(args.var_auto_lasso_coeff)), 'w' ) as f_coeff:
                    f_coeff.write( str(grp_lasso_coeff.item()) )

            else:
                with open( os.path.join(coeff_dir, str(args.var_auto_lasso_coeff)), 'r' ) as f_coeff:
                    for line in f_coeff:
                        grp_lasso_coeff = float(line)
            lasso_penalty = lasso_penalty * grp_lasso_coeff
        else:
            lasso_penalty = 0.

        loss += lasso_penalty

        lasso_time = time.time() - lasso_time_start

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        lasso_ratio.update(lasso_penalty / loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #loss.backward()
        optimizer.backward(loss)
        optimizer.step()

        # measure elapsed time
        #batch_time.update(time.time() - end - lasso_time)
        batch_time.update(time.time() - end - lasso_time - data_load_time)
        end = time.time()

        # plot progress
#        bar.suffix  = '({batch}/{size}) Data: {data:.5f}s | Batch: {bt:.5f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                    batch=batch_idx + 1,
#                    size=len(trainloader),
#                    data=data_time.avg,
#                    bt=batch_time.avg,
#                    total=bar.elapsed_td,
#                    eta=bar.eta_td,
#                    loss=losses.avg,
#                    top1=top1.avg,
#                    top5=top5.avg,
#                    )
#        bar.next()
#    bar.finish()
    epoch_time = batch_time.avg * len(trainloader)    # Time for total training dataset
    return (losses.avg, top1.avg, lasso_ratio.avg, epoch_time)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
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

        # plot progress
#        bar.suffix  = '({batch}/{size}) Data: {data:.5f}s | Batch: {bt:.5f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                    batch=batch_idx + 1,
#                    size=len(testloader),
#                    data=data_time.avg,
#                    bt=batch_time.avg,
#                    total=bar.elapsed_td,
#                    eta=bar.eta_td,
#                    loss=losses.avg,
#                    top1=top1.avg,
#                    top5=top5.avg,
#                    )
#        bar.next()
#    bar.finish()
    epoch_time = batch_time.avg * len(testloader)   # Time for total test dataset
    return (losses.avg, top1.avg, epoch_time)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state

    set_lr = args.lr
    for lr_decay in args.schedule:
        if epoch >= lr_decay:
            set_lr *= args.gamma
    state['lr'] = set_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

    #if epoch in args.schedule:
    #    state['lr'] *= args.gamma
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
