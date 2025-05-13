import torchvision
from torch.utils.tensorboard import SummaryWriter

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# target_transform 用于对 target(目标分类) 进行变换
# download 一直设置为 True 也没问题，并不会重复下载，目标路径有过了也会直接用
train_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=True, transform=data_transform, target_transform=None, download=True)
test_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=False, transform=data_transform, target_transform=None, download=True)

writer = SummaryWriter("logs")

for i in range(10):
    writer.add_image("train_set", train_set[i][0], i)

writer.close()