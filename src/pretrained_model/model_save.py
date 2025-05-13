import torch
import torchvision.models
# 需要从网上下载模态，其实是模型各层的参数
resnet18_true = torchvision.models.resnet18(pretrained=True, progress=True)
# 只会简单定义模型结构并初始化参数，不用下载
resnet18_false = torchvision.models.resnet18(pretrained=False)

# 方式1：保存模型结构和参数
torch.save(resnet18_false, "resnet18_false1.pth")
# 方式2：保存模型参数，以字典的形式 (官方推荐只保存参数，占用空间更小，且避免兼容性问题）
torch.save(resnet18_false.state_dict(), "resnet18_false2.pth")