# create by 敖鸥 at 2022/11/18
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_ds = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)

dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
