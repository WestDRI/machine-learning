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

# ** Plot one image with its pixel values

imgplot = plt.figure(figsize = (12,12))
sub = imgplot.add_subplot(111)
sub.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max() / 2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y].item(), 1) if img[x][y].item() !=0 else 0
        sub.annotate(str(val), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y].item() < thresh else 'black')

imgplot.show()
# imgplot.savefig('imgpx.png', bbox_inches='tight')

# * DataLoader

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=20, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=20, shuffle=False)

# * Plot a full batch of images with their labels

# get one batch of training images
dataiter = iter(train_loader)
batchimg, batchlabel = dataiter.next()

# plot the images and their label in that batch
batchplot = plt.figure()
for i in torch.arange(20):
    sub = batchplot.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    sub.imshow(torch.squeeze(batchimg[i]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    sub.set_title(str(batchlabel[i].item()))
batchplot.show()
# batchplot.savefig('batch.png', bbox_inches='tight')
