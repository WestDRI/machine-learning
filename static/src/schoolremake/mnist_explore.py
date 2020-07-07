import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# * Prepare data

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_data = datasets.MNIST(
    '~/parvus/pwg/wtm/tml/data',
    # '~/projects/def-sponsor00/data',
    train=True, download=True, transform=transform)

test_data = datasets.MNIST(
    '~/parvus/pwg/wtm/tml/data',
    # '~/projects/def-sponsor00/data',
    train=False, transform=transform)

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

# * Plot one image

img = train_data[0][0]
img = img.view(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
# plt.savefig('img.png', bbox_inches='tight')

# * DataLoader

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=20, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=20, shuffle=False)
