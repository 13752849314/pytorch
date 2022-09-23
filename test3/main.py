# create by 敖鸥 at 2022/9/23

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def main():
    batch_size = 32

    csf_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)

    csf_train = DataLoader(csf_train, batch_size=batch_size, shuffle=True)

    csf_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)

    csf_test = DataLoader(csf_test, batch_size=batch_size, shuffle=True)

    x, label = next(iter(csf_train))
    print('x:', x.shape, 'label:', label.shape)


if __name__ == '__main__':
    main()
