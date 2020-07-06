import torch

real = torch.rand(3, 8)
print(real)

predicted = torch.rand(3, 8, requires_grad=True)
print(predicted)

loss = (predicted - real).pow(2).sum()

with torch.no_grad():
    manual_gradient_predicted = 2.0 * (predicted - real)
print(manual_gradient_predicted)

loss.backward()
auto_gradient_predicted = predicted.grad
print(auto_gradient_predicted)

print(manual_gradient_predicted.eq(auto_gradient_predicted).all())
