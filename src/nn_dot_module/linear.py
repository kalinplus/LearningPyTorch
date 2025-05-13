import numpy as np
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

# 准备数据
test_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

# 定义网络
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=64 * 32 * 32 * 3, out_features=10)
    def forward(self, x):
        X = torch.flatten(x)
        x = self.linear(X)
        return x

linear = Linear()

# 将数据送入网络，获取输出（当然没有训练啦）
for i, (imgs, targets) in enumerate(test_loader):
    imgs = linear(imgs)
    print(np.argmax(imgs.detach().numpy()))