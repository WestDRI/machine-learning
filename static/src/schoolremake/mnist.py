import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# To run on GPU, replace 'cpu' by 'cuda'
device = torch.device('cpu')

train = datasets.MNIST(
    'projects/def-sponsor00/data',
    train = True,
    download = True,
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))

test = datasets.MNIST(
    'projects/def-sponsor00/data',
    train = False,
    download = False,
    transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)) ]))

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size = 4,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(
    test,
    batch_size = 4,
    shuffle = False)
