# Introduction

This repository containts the source codes and scripts for trianing a set of CNNs using PruneTrian, constiuous structured model pruning and network architecture reconfiguration (https://arxiv.org/pdf/1901.09290.pdf). We provide the code base to train ResNet and VGG models with different number of layers for CIFAR10, CIFAR100, and ResNet50 model for ImageNet.

| Dataset        | Model         | Policy             | Top1 Error | Link |
| -------------| ------------- |:------------------:|:----------:|:----:|
| CIFAR10      | ResNet32      | Dense baseline         |   24.0%    |      |   
| ^            | ResNet50      | Groups per layer   |   24.2%    |      |   
| ^            | ResNet50      | Channels per group |   23.8%    |      |   


# Training CIFAR10/100

We train using a mini-batch size of 128 distributed to 4 GPUs as the baseline. Please increase the learning rate to 0.1 when using mini-batch size 256.

