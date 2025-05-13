import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x += 1
        return x

model = Model()
x = torch.tensor(1.0)
print(f"input: {x}")
output = model(x)
print(f"output: {output}")