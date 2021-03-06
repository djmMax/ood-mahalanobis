"""
Adapted 2020 by Irena Gao 
Forked from Kimin Lee: https://github.com/pokaxpoka/deep_Mahalanobis_detector
Original code from: https://github.com/aaron-xichen/pytorch-playground

Functions to load datasets used in experiments. 
Model: resnet

"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# resnet transform
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

# VGG-16 Takes 224x224 images as input, so we resize all of them


MALARIA_TRANSFORMS = {
    'train': transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        # transforms.RandomResizedCrop(124),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.CenterCrop(124),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(124),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getMALARIA(batch_size, dataset_name, TF=None, data_root='/tmp/public_dataset/pytorch', train=True, val=True, test=False, **kwargs):
    print('data_root:', data_root)
    data_root = os.path.expanduser(os.path.join(data_root, dataset_name))
    #num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=data_root + '/train',
                transform=MALARIA_TRANSFORMS['train']),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
        print("Loaded {} / train".format(len(train_loader.dataset)))
        print(train_loader.dataset.classes)

    if val:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=data_root + '/val',
                transform=MALARIA_TRANSFORMS['test']),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(val_loader)
        print("Loaded {} / val".format(len(val_loader.dataset)))
        print(val_loader.dataset.classes)

    if test:
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=data_root + '/test',
                transform=MALARIA_TRANSFORMS['test']),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
        print("Loaded {} / test".format(len(test_loader.dataset)))
        print(test_loader.dataset.classes)

    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(data_type, batch_size, data_path='./data'):
    """
    loads in-dataset (both a train and test version)
    Args:
    - data_type = name of in-dataset
    - batch_size = size of chunks
    - data_path = path from pwd to data folder
    """
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=4)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=4)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=4)
    elif data_type == 'malaria':
        train_loader, test_loader = getMALARIA(
            batch_size=batch_size, dataset_name='malaria', data_root=data_path, num_workers=4)
    elif data_type == 'malaria2':
        train_loader, test_loader = getMALARIA(
            batch_size=batch_size, dataset_name='malaria2', data_root=data_path, num_workers=4)

    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, data_path='./data'):
    """
    loads out-dataset
    Args:
    - data_type = name of out-dataset
    - batch_size = size of chunks
    - data_path = path from pwd to data folder
    """
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(
            batch_size=batch_size, TF=TRANSFORM, data_root=data_path, num_workers=1)
    elif data_type == 'imagenet_resize':
        data_path = os.path.expanduser('./data/Imagenet_resize')
        testsetout = datasets.ImageFolder(data_path, transform=TRANSFORM)
        test_loader = torch.utils.data.DataLoader(
            testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        data_path = os.path.expanduser('./data/LSUN_resize')
        testsetout = datasets.ImageFolder(data_path, transform=TRANSFORM)
        test_loader = torch.utils.data.DataLoader(
            testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'malaria':
        train_loader, test_loader = getMALARIA(
            batch_size=batch_size, dataset_name='malaria', data_root=data_path, num_workers=4)
    elif data_type == 'malaria2':
        train_loader, test_loader = getMALARIA(
            batch_size=batch_size, dataset_name='malaria2', data_root=data_path, num_workers=4)
    return test_loader


def getAdversarialDataSet(attack_type, model, data_type, batch_size, adv_path='./output/adversarial/'):
    """
    loads attacked data generated by ADV_samples
    Args:
    - attack_type = name of attack
    - data_type = name of in-dataset
    - batch_size = size of chunks
    - adv_path = path from pwd to adv folder with .pth file
    Returns:
    - data in chunks
    - labels in chunks
    """
    adv = torch.load(adv_path + '/%s_%s/adv_data_%s_%s_%s.pth' %
                     (model, data_type, model, data_type, attack_type)).cuda()
    label = torch.load(adv_path + '/%s_%s/label_%s_%s_%s.pth' %
                       (model, data_type, model, data_type, attack_type)).cuda()
    return zip(adv.split(batch_size), label.split(batch_size))

if __name__ == '__main__':
    getMALARIA(100, data_root='./data')
