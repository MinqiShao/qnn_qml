import torch

a = torch.tensor([1, 2, 0, 1, 3, 0])
b = torch.tensor([0, 1])
print(len(torch.where(torch.logical_or(a == b[0], a == b[1]))[0]))

