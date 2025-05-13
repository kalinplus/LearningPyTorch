import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader

# 准备数据
test_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

# 定义 CIFAR10 模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

model = Model()
for i, (imgs, targets) in enumerate(test_loader):
    # imgs: Tensor(64, 3, 32, 32)
    imgs = model(imgs)
    print(imgs.shape)