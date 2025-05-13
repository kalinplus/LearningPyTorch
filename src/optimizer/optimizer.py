import torch.optim
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
loss_fn = nn.CrossEntropyLoss()  # 其实是类对象
# 使用优化器，训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epoch = 20

for epoch in range(num_epoch):
    print(f"Epoch: {epoch+1}")
    model.train()  # 训练模式
    average_loss = 0.0  # 展示平均损失
    for i, (imgs, targets) in enumerate(test_loader):
        # 清除上一次循环的梯度，一定要写！
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = loss_fn(predictions, targets)
        average_loss += loss.item()
        loss.backward()  # 反向传播算出梯度，存储在对应的可训练参数中
        optimizer.step()  # 优化器更新参数
    average_loss /= len(test_loader)
    print(f"Average Loss: {average_loss}")
