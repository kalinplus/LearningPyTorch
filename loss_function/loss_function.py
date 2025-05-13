import torch
import torch.nn as nn

# 准备数据
output = torch.Tensor([1, 2, 3])
target = torch.Tensor([1, 3, 5])

# 准备损失函数类对象
loss_l1 = nn.L1Loss(reduction='mean')
loss_l2 = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()

# 计算 l1 和 l2 孙树
# 我想要 output/target 的维度为 (N, C, ...)
output = torch.reshape(output, (1, 1, -1))
target = torch.reshape(target, (1, 1, -1))
print(f"l1 loss: {loss_l1(output, target)}")
print(f"l2 loss: {loss_l2(output, target)}")

# 计算 交叉熵损失
# output 的维度应为 (N, C)，现在里面的 3 个数被视为 C 的得分
# target 为类别索引，不是 one-hot 向量
output = torch.tensor([[0.1, 0.8, 0.2]])
target = torch.tensor([1], dtype=torch.long)
print(f"cross entropy loss: {loss_ce(output, target)}")