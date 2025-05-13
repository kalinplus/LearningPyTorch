import torch
import torchvision

# torch 中包含 resnet18 类的定义，所以没问题
# 但是如果是自定义的模型结构，就需要在本文件中引入/定义对应类，再加载，否则使用时会报未定义的错
# 加载方式1
resnet18_false1 = torch.load("resnet18_false1.pth")
print(resnet18_false1)
# 加载方式2
resnet18_false2 = torchvision.models.resnet18(pretrained=False)
resnet18_false2.load_state_dict(torch.load("resnet18_false2.pth"))
print(resnet18_false2)