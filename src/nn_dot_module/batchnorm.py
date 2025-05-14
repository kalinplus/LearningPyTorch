import torch
from torch import nn

# 准备数据
# 常见于 [N, C, H, W] 维度
input4d = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0]])
input4d = torch.reshape(input4d, (1, 1, 2, 2))
# 常见于 [N, C] 维度
# 训练模式下要求 batch_size (N) 至少为 2
input2d = torch.tensor([[1.0, 3.0, 5.0],
                        [2.0, 4.0, 7.0]])
# 定义网络结构
class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建这两个类对象时，需要的参数 num_features 都是 通道数
        self.BatchNorm1 = nn.BatchNorm1d(num_features=3)
        self.BatchNorm2 = nn.BatchNorm2d(num_features=1)
    def forward(self, x):
        if len(x.shape) == 4:
            return self.BatchNorm2(x)
        elif len(x.shape) == 2 or len(x.shape) == 3:
            return self.BatchNorm1(x)
        else:
            return None

batchnorm = BatchNorm()
print(f"4d 输入进入 BatchNorm2d 前：{input4d}")
print(f"4d 输入经过 BatchNorm2d 后：{batchnorm(input4d)}")

print(f"2d 输入进入 BatchNorm1d 前：{input2d}")
print(f"2d 输入经过 BatchNorm1d 后：{batchnorm(input2d)}")