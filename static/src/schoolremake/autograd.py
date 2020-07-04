import torch
import random

y = torch.tensor(random.choices(range(10), k = 50),
                 dtype = torch.float)
print(y)

y_pred = torch.tensor(random.choices(range(10), k = 50),
                      dtype = torch.float, requires_grad = True)
print(y_pred)

loss = (y_pred - y).pow(2).sum()

loss.backward()
print(y_pred.grad)

manual_grad_y_pred = 2.0 * (y_pred - y)
print(manual_grad_y_pred)

print(manual_grad_y_pred.eq(y_pred.grad).all())
