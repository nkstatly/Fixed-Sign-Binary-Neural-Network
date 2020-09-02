import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
    
def get_loader_cifar10(args, kwargs=None):
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]
    loader_train = None
    
    if kwargs == None:
        kwargs = {
            'num_workers': args.n_threads,
            'pin_memory': True
        }

    if args.is_train:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())
        
        transform_train = transforms.Compose(transform_list)

        loader_train = DataLoader(
            datasets.CIFAR10(
                root=args.dir_data,
                train=True,
                download=True,
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.CIFAR10(
            root=args.dir_data,
            train=False,
            download=True,
            transform=transform_test),
        batch_size=500, shuffle=False,)

    return loader_train, loader_test

def get_loader_imagenet(dir_data='/root/datas/', no_flip=False, batch_size=128): 
    """
    Follow https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset to prepare the dataset.
    """
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    loader_train = None
    kwargs = {
        'num_workers': 40,
        'pin_memory': True
    }

    transform_list = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)]

    if not no_flip:
        transform_list.remove(transform_list[1])

    transform_train = transforms.Compose(transform_list)

    loader_train = DataLoader(
        datasets.ImageFolder(
            root=os.path.join(dir_data, 'Imagenet2012', 'train'),
            transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs
    )


    transform_list = [transforms.Resize(256)]
    batch_test = 128

    transform_list.append(transforms.CenterCrop(224))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(norm_mean, norm_std))

    transform_test = transforms.Compose(transform_list)

    loader_test = DataLoader(
        datasets.ImageFolder(
            root=os.path.join(dir_data, 'Imagenet2012', 'val'),
            transform=transform_test),
        batch_size=batch_test, shuffle=False, **kwargs
    )

    return loader_train, loader_test
