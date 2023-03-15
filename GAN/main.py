# create by 敖鸥 at 2022/11/18
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_ds = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)

dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(-1, 1, 28, 28)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.main(x)
        return out


def gen_img_plot(model, test_input):
    pred = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((pred[i] + 1) / 2)
        plt.axis('off')
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)

g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)

loss_fn = nn.BCELoss().to(device)

test_input = torch.randn(16, 100, device=device)

D_loss = []
G_loss = []

for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)

    for step, (img, label) in enumerate(dataloader):
        img1 = img.to(device)
        size = img1.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()
        real_output = dis(img1)  # 判别器输入真实图片
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output, device=device))
        d_real_loss.backward()

        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())  # 判别器输入生成图片
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output1 = dis(gen_img)
        g_loss = loss_fn(fake_output1, torch.ones_like(fake_output1, device=device))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)

        print('epoch', epoch)
        print('d_loss', d_epoch_loss)
        print('g_loss', g_epoch_loss)

        gen_img_plot(gen, test_input)
