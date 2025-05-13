import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class MyPooling(nn.Module):
    def __init__(self):
        super(MyPooling, self).__init__()
        # self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)
        # stride 默认与 kernel_size 相同，无需额外设置
        self.pool = nn.MaxPool2d(kernel_size=2, padding=0)
    def forward(self, x):
        # x = self.conv(x)
        x = self.pool(x)
        return x

conv = MyPooling()
test_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)
writer = SummaryWriter("logs")

for i, (imgs, targets) in enumerate(test_loader):
    imgs = conv(imgs)
    # print(imgs.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("pooling", imgs, i)

writer.close()