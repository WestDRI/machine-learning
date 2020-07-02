import torch

torch.rand(2).item()

x = torch.rand(2, 3)
print(x)

torch.rand(1)
torch.rand(1, 1).dim()

torch.rand(1).item()

torch.rand(1, 1)

torch.rand(1, 1, 1)

x = torch.rand(2)
y = torch.rand(2)

torch.add(x, y)

x.add_(y)

torch.rand(3, 1)

torch.rand(2, 1, 5)
torch.rand(2, 2, 5)

torch.rand(2, 2, 5)
torch.rand(1, 1, 5)
torch.rand(1, 1, 5, 1)
torch.rand(2, 3, 5, 2)
torch.rand(2, 3, 5, 2, 4)
torch.rand(3, 5, 4, 2, 1).dim()

torch.rand(3, 5, 4, 2, 1).size()
