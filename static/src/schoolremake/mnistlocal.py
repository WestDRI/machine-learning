import torch
from torchvision import datasets, transforms

# To run on GPU, replace 'cpu' by 'cuda'
device = torch.device('cpu')

# * Prepare data

train = datasets.MNIST(
    './data',
    train = True,
    download = True,
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))

test = datasets.MNIST(
    './data',
    train = False,
    download = False,
    transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))

# * Explore data

len(train)

train[0]
len(train[0])
type(train[0])

train[0][0]
type(train[0][0])

train[0][1]
type(train[0][1])

train[0][0].size()

train[0][0][0]
train[0][0][0][0]
train[0][0][0][0][0]

# * Print one image

from matplotlib import pyplot as plt

img = train[0][0]
img = img.view(28, 28)
plt.imshow(img, cmap='gray')
plt.savefig('img.png')

# * DataLoader

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size = 4,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(
    test,
    batch_size = 4,
    shuffle = False)
