# create by 敖鸥 at 2023/4/3
import torch

path = r'D:\桌面\models\segmentation_OST_bic.pth'
model = torch.load(path)
print(model)
