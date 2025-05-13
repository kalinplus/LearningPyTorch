import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

tensor = torch.Tensor([[1, -2],
                       [-0.8, 5]])
# 参考 非线性激活 模块的文档，它们需要输入有一个 batch_size (N)，后面的维度其实无所谓
tensor = torch.reshape(tensor, (1, 1, 2, 2))

test_set = torchvision.datasets.CIFAR10(root='./dataset/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

class MyActivation(nn.Module):
    def __init__(self):
        super(MyActivation, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # output = self.relu(x)
        output = self.tanh(x)
        return output

act = MyActivation()
# output = act(tensor)
# print(f"input: {tensor}")
# print(f"output: {output}")
writer = SummaryWriter("logs")
for i, (imgs, targets) in enumerate(test_loader):
    imgs = act(imgs)
    writer.add_images("tanh", imgs, i)
writer.close()