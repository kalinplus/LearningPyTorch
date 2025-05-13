import torch
import torchvision.models
# 需要从网上下载模态，其实是模型各层的参数
resnet18_true = torchvision.models.resnet18(pretrained=True, progress=True)
# 只会简单定义模型结构并初始化参数，不用下载
resnet18_false = torchvision.models.resnet18(pretrained=False)

# 现在我想要修改 Resnet18 模型，让它的输出层输出 10 个类别，而不是 1000 个类别
# 替换最后的 FC 层
resnet18_true.fc = torch.nn.Linear(512, 10)
print(resnet18_true)