# create by 敖鸥 at 2022/9/23
import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32 ,32] => [b, 6, ]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        )
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        batsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batsz, 16 * 5 * 5)
        logits = self.fc_unit(x)

        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits)
        return logits