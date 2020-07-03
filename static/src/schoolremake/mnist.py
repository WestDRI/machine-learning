import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# To run on GPU, replace 'cpu' by 'cuda'
device = torch.device('cpu')

# * Prepare data

train = datasets.MNIST(
    './data',
    # '~/projects/def-sponsor00/data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

test = datasets.MNIST(
    './data',
    # '~/projects/def-sponsor00/data',
    train = False,
    download = False,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

# * Explore data

print(len(train))

print(train[0])
print(len(train[0]))
print(type(train[0]))

print(train[0][0])
print(type(train[0][0]))

print(train[0][1])
print(type(train[0][1]))

print(train[0][0].size())

print(train[0][0][0])
print(train[0][0][0][0])
print(train[0][0][0][0][0])

print(train.data)
print(train.data.size())

print(train.targets)
print(train.targets.size())

# * Print one image

print(img = train[0][0])
print(img = img.view(28, 28))
plt.imshow(img, cmap='gray')
# plt.show()
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
