import torch

x = torch.rand(2, 3)
print(x)

print(torch.rand(1))
print(torch.rand(1, 1).dim())

print(torch.rand(1).item())

print(torch.rand(1, 1))

print(torch.rand(1, 1, 1))

x = torch.rand(2)
y = torch.rand(2)

print(torch.add(x, y))

print(x.add_(y))

print(torch.rand(3, 1))

print(torch.rand(2, 1, 5))
print(torch.rand(2, 2, 5))

print(torch.rand(2, 2, 5))
print(torch.rand(1, 1, 5))
print(torch.rand(1, 1, 5, 1))
print(torch.rand(2, 3, 5, 2))
print(torch.rand(2, 3, 5, 2, 4))
print(torch.rand(3, 5, 4, 2, 1).dim())

print(torch.rand(3, 5, 4, 2, 1).size())
