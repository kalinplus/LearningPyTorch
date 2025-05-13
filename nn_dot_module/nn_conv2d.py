import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class MyConv(nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()
        # kernel 只要指定 kernel_size 即可，内部参数值按一定的分布初始化
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        # 输出可以使用公式计算，或者使用 print、debug 查看
        x = torch.reshape(x, (-1, 3, 30, 30))
        return x

conv = MyConv()
test_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)
writer = SummaryWriter("logs")

for i, (imgs, targets) in enumerate(test_loader):
    imgs = conv(imgs)
    # print(imgs.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("conv2d", imgs, i)

writer.close()