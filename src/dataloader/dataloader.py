import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='./dataset/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
# Windows 下 num_workers 好像有问题，如果出现 Pipe Broken, 可以尝试将其值设置为 0
test_dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

writer = SummaryWriter("logs")

for epoch in range(2):
    for step, (imgs, targets) in enumerate(test_dataloader):
        writer.add_images(f"test_set_epoch{epoch}", imgs, step)
        # print(imgs.shape)
writer.close()