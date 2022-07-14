---
title: Building a neural network
description: Zoom
colordes: "#e86e0a"
slug: 13_building_nn
weight: 13
execute:
  error: true
format: hugo
jupyter: python3
---



Key to creating neural networks in PyTorch is the `torch.nn` package which contains the `nn.Module` and a `forward` method which returns an output from some input.

Let's build a neural network to classify the MNIST.

## Load packages

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Define the architecture of the network

First, we need to define the architecture of the network.

There are many types of architectures. For images, CNN are well suited.

In Python, you can define a subclass of an existing class with:

``` python
class YourSubclass(BaseClass):
    <definition of your subclass>        
```

Your subclass is derived from the base class and inherits its properties.

PyTorch contains the class `torch.nn.Module` which is used as the base class when defining a neural network.

``` python
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      
      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features.
      # Convolution adds each pixel of an image to its neighbours,
      # weighted by the kernel (a small matrix).
      # Here, the kernel is square and of size 3*3
      # Convolution helps to extract features from the input
      # (e.g. edge detection, blurriness, sharpeness...)
      self.conv1 = nn.Conv2d(1, 32, 3)
      # Second 2D convolutional layer, taking in the 32 input channels,
      # outputting 64 convolutional features, with a kernel size of 3*3
      self.conv2 = nn.Conv2d(32, 64, 3)

      # Dropouts randomly blocks a fraction of the neurons during training
      # This is a regularization technique which prevents overfitting
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # First fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 10)
```

## Set the flow of data through the network

The feed-forward algorithm is defined by the `forward` function.

``` python
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3)
      self.conv2 = nn.Conv2d(32, 64, 3)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents the data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output
```

Let's create an instance of `Net` and print its structure:

``` python
net = Net()
print(net)
```

    Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout2d(p=0.25, inplace=False)
      (dropout2): Dropout2d(p=0.5, inplace=False)
      (fc1): Linear(in_features=9216, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )

## Comments & questions
