from __future__ import print_function
import torch

# for those without a GPU
device = torch.device('cpu')

# for those with a CUDA-enabled GPU
# device = torch.device('cuda')

# N is the batch size
# D_in is the dimension of the input layer
# H is the dimension of the hidden layer
# D_out is the dimension of the output layer
N, D_in, H, D_out = 64, 1000, 100, 10

# Create 2d random input and output data
# Similar to numpy.random.randn()
x = torch.randn(N, D_in, device = device)
y = torch.randn(N, D_out, device = device)

# Randomly initialize the weights
# from the input layer to the hidden layer
w1 = torch.randn(D_in, H, device = device)

# Randomly initialize the weights
# from the hidden layer to the output layer
w2 = torch.randn(H, D_out, device = device)

learning_rate = 1e-6

# We are going over 500 epochs
for t in range(500):

  # Forward pass: compute predicted y
  h = x.mm(w1)
  h_relu = h.clamp(min=0)
  y_pred = h_relu.mm(w2)

  # Compute and print loss; loss is a scalar, and is stored in a PyTorch Tensor
  # of shape (); we can get its value as a Python number with loss.item().
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.item())

  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.t().mm(grad_y_pred)
  grad_h_relu = grad_y_pred.mm(w2.t())
  grad_h = grad_h_relu.clone()
  grad_h[h < 0] = 0
  grad_w1 = x.t().mm(grad_h)

  # Update weights using gradient descent
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
