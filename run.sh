#python run-script.py --data-train /mnt/dataset/imagenet-data/raw-data/train --data-val /mnt/dataset/imagenet-data/raw-data/validation --dataset imagenet --model resnet50 --num-gpus 2
#python run-script.py --data-train /mnt/dataset/imagenet-data/raw-data/train --data-val /mnt/dataset/imagenet-data/raw-data/validation --dataset imagenet --model vgg16 --num-gpus 2
python run-script.py --data-train /mnt/dataset/imagenet-data/raw-data/train --data-val /mnt/dataset/imagenet-data/raw-data/validation --dataset cifar10 --model resnet32 --num-gpus 2
