# training full-precision ResNet-18 on CIFAR10
#python main.py --is_train --model resnet --depth 18 --dataset cifar10 --origin

# training ScaleNet with ResNet-18 on CIFAR10
#python main.py --is_train --model resnet --depth 18 --dataset cifar10 --input_bits 32 --scale_bits 16

# training ScaleNet with ResNet-18 on ImageNet
#python main.py --is_train --model resnet --depth 18 --dataset imagenet --input_bits 32 --scale_bits 16
