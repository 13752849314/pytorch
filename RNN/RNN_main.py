# create by 敖鸥 at 2022/10/28
import torch
import numpy as np
from torch import nn, optim

from RNN import SimpleRNN

input_size = 1
output_size = 1
hidden_size = 12
batch_size = 1
num_layers = 1
max_iter = 1000
num_time_steps = 50

model = SimpleRNN(input_size, hidden_size, output_size, num_layers=num_layers)
print(model)
criteon = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

h_prev = torch.zeros(num_layers * batch_size, input_size, hidden_size)

for epoch in range(max_iter):
    model.train()
    start = np.random.randint(10, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(batch_size, num_time_steps - 1, output_size)
    y = torch.tensor(data[1:]).float().view(batch_size, num_time_steps - 1, output_size)

    output, h_prev = model(x, h_prev)
    h_prev = h_prev.detach()

    loss = criteon(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch=', epoch, 'loss=', loss.item())

model.eval()
predictions = []
pred_start = 0
inp = torch.tensor([[pred_start]]).float()
p_num = num_time_steps
for _ in range(p_num):
    inp = inp.view(1, 1, 1)
    pred, h_prev = model(inp, h_prev)
    inp = pred
    predictions.append(pred.detach().numpy().ravel()[0])

print(predictions)

import matplotlib.pyplot as plt

x = np.linspace(pred_start, pred_start + 10, num_time_steps)
y = np.sin(x)
plt.plot(x, y, 'r-', x, predictions, 'g-')
plt.show()
