import os
import numpy as np
import torch.nn
import torchvision.datasets
from torch.nn import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import cifar10

# 准备数据集
totensor = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=True, transform=totensor, download=True)
test_set = torchvision.datasets.CIFAR10(root='../dataset/data', train=False, transform=totensor, download=True)

# 定义超参数
learning_rate = 5e-4
weight_decay = 1e-4
num_epoch = 10
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 “cuda: 0" 对我电脑应该效果一样

# 准备 dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

# 定义模型架构
model = cifar10()
model = model.to(device)

# 检验模型结构是否正确
# for (imgs, targets) in train_loader:
#     print(model(imgs).shape)  # 输出 [64, 10]，说明正确

# 定义损失函数
loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 定义 SummaryWriter
writer = SummaryWriter('logs')
total_train_step = 0
total_val_step = 0

# 用于计算何时保存模型（参数）
best_acc = 0.0

# 开始训练！先搭建基本框架，再逐步添加更多功能
# 1. 打印 loss。看起来没问题，能 train
# 2. 计算并打印训练阶段预测准确率
# 3. 打印 1 个 epoch 的平均损失
# 4. 添加验证集，并支持上述功能 1 2 3
# 5. 在 tensorboard 绘制 训练 和 验证 曲线
# 6. 保存模型，做到如果在验证集上，准确率超过最优模型，则保存
for epoch in range(num_epoch):
    print(f"------------ Epoch {epoch} ------------")
    # 训练部分
    model.train()
    acc_list = []
    loss_list = []
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        # loss, acc 实际上都是一个 batch 的平均值
        loss = loss_fn(preds, targets)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        labels = torch.argmax(preds, dim=1)
        acc = torch.sum(labels == targets) / batch_size
        acc_list.append(acc)
        # 绘制曲线图
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar('train_loss', loss.item(), total_train_step)
            writer.add_scalar('train_acc', acc, total_train_step)
        # print(f"第 {i} 个 batch | " +
        #       f"loss = {loss.item()} | " +
        #       f"Train Acc = {acc_rate}")
    print("训练集")
    print(f"loss = {np.mean(loss_list)}")
    print(f"Acc = {np.mean(acc_list)}")
    print()

    # 验证部分
    model.eval()
    with torch.no_grad():
        acc_list = []
        loss_list = []
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            # 交叉熵损失就是需要 一个概率数组（输入）和一个目标标签数值（目标）
            # 所以不能直接求预测的 label
            loss = loss_fn(preds, targets)
            loss_list.append(loss.item())
            labels = torch.argmax(preds, dim=1)
            acc = torch.sum(labels == targets) / batch_size
            acc_list.append(acc)
            # 绘制曲线图
            total_val_step += 1
            if total_val_step % 100 == 0:
                writer.add_scalar('val_loss', loss.item(), total_val_step)
                writer.add_scalar('val_acc', acc, total_val_step)
        print("验证集")
        print(f"loss = {np.mean(loss_list)}")
        mean_acc = np.mean(acc_list)
        print(f"Acc = {mean_acc}", end=" ")
        # 保存模型
        if mean_acc > best_acc:
            best_acc = mean_acc
            if not os.path.exists('pth'):
                os.mkdir('pth')
            torch.save(model.state_dict(), f"./pth/model_CIFAR10_epoch{epoch}_acc{mean_acc}.pth")
            print(f"<-- 当前最好模型，保存至 ./pth/model_CIFAR10_epoch{epoch}_acc{mean_acc}.pth")
        print()