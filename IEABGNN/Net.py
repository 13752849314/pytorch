# create by 敖鸥 at 2022/11/14
import torch
from torch import nn
from torch.nn import functional as fun


class ResNet_Block1(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super(ResNet_Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion)
            )

    def forward(self, x):
        x1 = fun.relu(self.bn1(self.conv1(x)))
        x1 = self.bn2(self.conv2(x1))
        x1 = x1 + self.shortcut(x)
        out = fun.relu(x1)
        return out


class ResNet_Block2(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super(ResNet_Block2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion)
            )

    def forward(self, x):
        x1 = fun.relu(self.bn1(self.conv1(x)))
        x1 = fun.relu(self.bn2(self.conv2(x1)))
        x1 = self.bn3(self.conv3(x1))
        x1 = x1 + self.shortcut(x)
        out = fun.relu(x1)
        return out


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)


class TransConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=1, padding=0):
        super(TransConv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv1(x)


class Gen(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Gen, self).__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv(out_ch, out_ch * 2, stride=2)
        self.conv3 = Conv(out_ch * 2, out_ch * 4, stride=2)

        self.block1 = ResNet_Block1(out_ch * 4, out_ch * 4)
        self.block2 = ResNet_Block1(out_ch * 4, out_ch * 4)
        self.block3 = ResNet_Block1(out_ch * 4, out_ch * 4)
        self.block4 = ResNet_Block1(out_ch * 4, out_ch * 4)
        self.block5 = ResNet_Block1(out_ch * 4, out_ch * 4)
        self.block6 = ResNet_Block1(out_ch * 4, out_ch * 4)

        self.trans1 = TransConv(out_ch * 4, out_ch * 2, stride=2, padding=1)
        self.trans2 = TransConv(out_ch * 2, out_ch, stride=2, padding=1)
        self.trans3 = TransConv(out_ch, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.trans1(x)
        x = self.trans2(x)
        x = self.trans3(x)

        return x


if __name__ == '__main__':
    m = ResNet_Block1(3, 32, 2)
    print(m)
    t = torch.randn(12, 3, 256, 256)
    o = m(t)
    print(o.shape)

    m1 = ResNet_Block2(3, 32, 2)
    print(m1)
    o1 = m1(t)
    print(o1.shape)

    gen = Gen(3, 64)
    print(gen)
    o2 = gen(t)
    print(o2.shape)
