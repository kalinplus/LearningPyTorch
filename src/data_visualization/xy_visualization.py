from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("log")
"""
通过 add_scalar 向 tensorboard 中写入 y-x 坐标图
"""
for i in range(100):
    writer.add_scalar("y=x^2", i*i, i)  # 查看文档可知，add_scalar函数的参数分别为：标签、y轴、x轴
writer.close()