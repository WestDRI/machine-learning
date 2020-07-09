# heavily inspired by https://github.com/iam-mhaseeb/Multi-Layer-Perceptron-MNIST-with-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_data = datasets.MNIST(
    '~/parvus/pwg/wtm/tml/data',
    # '$SLURM_TMPDIR/data',
    train=True, download=True, transform=transform)

test_data = datasets.MNIST(
    '~/parvus/pwg/wtm/tml/data',
    # '$SLURM_TMPDIR/data',
    train=False, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=20, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=20, shuffle=False)

# define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        return x

# initialize the network
model = Net()
print(model)

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# number of epochs to train the model
epochs = 20

# prep model for training
model.train()

for epoch in range(epochs):

    # monitor training loss
    train_loss = 0.0

    # train the model #
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)

    print('Epoch: {} \ttraining loss: {:.6f}'.format(
        epoch + 1,
        train_loss
        ))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = torch.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss / len(test_loader.dataset)
print('Test loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            torch.sum(class_correct[i]), torch.sum(class_total[i])))
    else:
        print('Test accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest accuracy (overall): %2d%% (%2d/%2d)' % (
    100. * torch.sum(class_correct) / torch.sum(class_total),
    torch.sum(class_correct), torch.sum(class_total)))

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for i in torch.arange(20):
    sub = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    sub.imshow(torch.squeeze(images[i]), cmap='gray')
    sub.set_title("{} ({})".format(str(preds[i].item()), str(labels[i].item())),
                 color=("green" if preds[i] == labels[i] else "red"))
