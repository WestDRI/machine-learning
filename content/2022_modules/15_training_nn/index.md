---
title: Training a model
description: Zoom
colordes: "#e86e0a"
slug: 15_training_nn
weight: 15
execute:
  error: true
format: hugo
jupyter: python3
---



## Packages, DataLoaders, model

Let's quickly run some code on the steps that we are now familiar with:

-   load the needed packages,
-   get the data,
-   create data loaders for training and testing,
-   define our model:

``` python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="~/projects/def-sponsor00/data/",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root="~/projects/def-sponsor00/data/",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

train_dataloader = DataLoader(training_data, batch_size=10)
test_dataloader = DataLoader(test_data, batch_size=10)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = Net()
```

## Hyperparameters

There is a set of specifications in the process of deep learning that we haven't talked about yet: hyperparameters.

While the learning parameters of a model (weights and biases) are the values that get adjusted through training (and they will become part of the final program, along with the model architecture, once training is over), hyperparameters control the training process.

They include:

-   batch size: number of samples passed through the model before the parameters are updated,
-   number of epochs: number iterations,
-   learning rate: size of the incremental changes to model parameters at each iteration. Smaller values yield slow learning speed, while large values may miss minima.

Let's define them here:

``` python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## Define the loss function

To assess the predicted outputs of our model against the true values from the labels, we also need a loss function (e.g. mean square error for regressions: `nn.MSELoss` or negative log likelihood for classification: `nn.NLLLoss`)

The machine learning literature is rich in information about various loss functions.

Here is an example with `nn.CrossEntropyLoss` which combines `nn.LogSoftmax` and `nn.NLLLoss`:

``` python
loss_fn = nn.CrossEntropyLoss()
```

## Initialize the optimizer

The optimization algorithm determines how the model parameters get adjusted at each iteration.

There are many optimizers and you need to search in the literature which one performs best for your time of model and data.

Below is an example with stochastic gradient descent:

``` python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

{{<notes>}}
- lr is the learning rate
- momentum is a method increasing convergence rate and reducing oscillation for SDG
{{</notes>}}

## Define the train and test loops

Finally, we need to define the train and test loops.

The train loop:

-   gets a batch of training data from the DataLoader,
-   resets the gradients of model parameters with `optimizer.zero_grad()`,
-   calculates predictions from the model for an input batch,
-   calculates the loss for that set of predictions vs. the labels on the dataset,
-   calculates the backward gradients over the learning parameters (that's the backpropagation) with `loss.backward()`,
-   adjusts the parameters by the gradients collected in the backward pass with `optimizer.step()`.

The test loop evaluates the model's performance against the test data.

``` python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

## Train

To train our model, we just run the loop over the epochs:

``` python
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Training completed")
```

    Epoch 1
    -------------------------------
    loss: 2.313032  [    0/60000]
    loss: 2.189277  [ 1000/60000]

    loss: 1.879373  [ 2000/60000]
    loss: 1.599335  [ 3000/60000]

    loss: 1.177001  [ 4000/60000]
    loss: 1.051830  [ 5000/60000]

    loss: 1.371127  [ 6000/60000]
    loss: 0.810428  [ 7000/60000]

    loss: 0.991967  [ 8000/60000]
    loss: 0.956140  [ 9000/60000]

    loss: 0.602727  [10000/60000]

    loss: 0.834871  [11000/60000]
    loss: 1.087197  [12000/60000]

    loss: 0.664605  [13000/60000]
    loss: 0.509106  [14000/60000]

    loss: 0.678273  [15000/60000]
    loss: 0.966540  [16000/60000]

    loss: 0.344180  [17000/60000]
    loss: 1.553607  [18000/60000]

    loss: 0.733053  [19000/60000]

    loss: 0.433200  [20000/60000]
    loss: 0.181992  [21000/60000]

    loss: 0.385904  [22000/60000]
    loss: 0.749857  [23000/60000]

    loss: 0.647050  [24000/60000]
    loss: 0.483924  [25000/60000]

    loss: 0.414128  [26000/60000]
    loss: 0.153665  [27000/60000]

    loss: 1.086485  [28000/60000]

    loss: 0.680597  [29000/60000]

    loss: 0.526780  [30000/60000]

    loss: 0.193959  [31000/60000]

    loss: 0.623465  [32000/60000]

    loss: 0.700794  [33000/60000]

    loss: 0.854201  [34000/60000]

    loss: 0.450702  [35000/60000]

    loss: 0.265808  [36000/60000]

    loss: 0.762630  [37000/60000]

    loss: 0.481670  [38000/60000]

    loss: 0.584916  [39000/60000]
    loss: 0.938548  [40000/60000]

    loss: 0.349947  [41000/60000]

    loss: 0.171913  [42000/60000]

    loss: 0.693459  [43000/60000]

    loss: 0.422915  [44000/60000]
    loss: 0.673171  [45000/60000]

    loss: 0.535404  [46000/60000]

    loss: 0.304551  [47000/60000]

    loss: 0.497037  [48000/60000]
    loss: 0.591269  [49000/60000]

    loss: 0.110107  [50000/60000]
    loss: 0.948390  [51000/60000]

    loss: 0.234284  [52000/60000]
    loss: 0.696888  [53000/60000]

    loss: 0.449288  [54000/60000]
    loss: 0.482404  [55000/60000]

    loss: 0.345427  [56000/60000]
    loss: 0.668137  [57000/60000]

    loss: 0.682948  [58000/60000]
    loss: 0.451150  [59000/60000]

    Test Error: 
     Accuracy: 103.2%, Avg loss: 0.474787 

    Epoch 2
    -------------------------------
    loss: 0.660638  [    0/60000]

    loss: 0.382847  [ 1000/60000]

    loss: 0.262217  [ 2000/60000]

    loss: 0.524995  [ 3000/60000]
    loss: 0.102952  [ 4000/60000]

    loss: 0.266365  [ 5000/60000]

    loss: 0.679761  [ 6000/60000]
    loss: 0.511284  [ 7000/60000]

    loss: 0.341274  [ 8000/60000]
    loss: 0.482237  [ 9000/60000]

    loss: 0.273982  [10000/60000]

    loss: 0.410666  [11000/60000]

    loss: 0.658335  [12000/60000]

    loss: 0.463334  [13000/60000]
    loss: 0.156792  [14000/60000]

    loss: 0.272522  [15000/60000]
    loss: 0.972761  [16000/60000]

    loss: 0.088276  [17000/60000]

    loss: 1.143931  [18000/60000]
    loss: 0.760979  [19000/60000]

    loss: 0.209964  [20000/60000]
    loss: 0.071765  [21000/60000]

    loss: 0.335469  [22000/60000]
    loss: 0.462239  [23000/60000]

    loss: 0.428549  [24000/60000]
    loss: 0.308786  [25000/60000]

    loss: 0.259935  [26000/60000]
    loss: 0.066200  [27000/60000]

    loss: 1.080371  [28000/60000]
    loss: 0.396030  [29000/60000]

    loss: 0.286784  [30000/60000]
    loss: 0.121891  [31000/60000]

    loss: 0.478226  [32000/60000]

    loss: 0.549788  [33000/60000]

    loss: 0.688560  [34000/60000]

    loss: 0.372566  [35000/60000]

    loss: 0.161006  [36000/60000]

    loss: 0.717515  [37000/60000]
    loss: 0.351387  [38000/60000]

    loss: 0.595150  [39000/60000]
    loss: 0.991084  [40000/60000]

    loss: 0.168918  [41000/60000]
    loss: 0.084852  [42000/60000]

    loss: 0.554675  [43000/60000]
    loss: 0.381406  [44000/60000]

    loss: 0.555818  [45000/60000]
    loss: 0.467898  [46000/60000]

    loss: 0.164762  [47000/60000]

    loss: 0.360958  [48000/60000]
    loss: 0.521409  [49000/60000]

    loss: 0.109901  [50000/60000]
    loss: 0.875834  [51000/60000]

    loss: 0.238080  [52000/60000]
    loss: 0.541535  [53000/60000]

    loss: 0.375447  [54000/60000]
    loss: 0.425235  [55000/60000]

    loss: 0.266625  [56000/60000]
    loss: 0.510005  [57000/60000]

    loss: 0.637097  [58000/60000]
    loss: 0.401188  [59000/60000]

    Test Error: 
     Accuracy: 108.1%, Avg loss: 0.412004 

    Epoch 3
    -------------------------------
    loss: 0.525211  [    0/60000]

    loss: 0.402093  [ 1000/60000]

    loss: 0.228855  [ 2000/60000]

    loss: 0.462229  [ 3000/60000]

    loss: 0.068593  [ 4000/60000]

    loss: 0.216395  [ 5000/60000]

    loss: 0.600057  [ 6000/60000]

    loss: 0.452461  [ 7000/60000]

    loss: 0.290583  [ 8000/60000]
    loss: 0.332457  [ 9000/60000]

    loss: 0.215503  [10000/60000]

    loss: 0.276737  [11000/60000]
    loss: 0.543060  [12000/60000]

    loss: 0.367736  [13000/60000]
    loss: 0.120917  [14000/60000]

    loss: 0.182735  [15000/60000]
    loss: 1.020491  [16000/60000]

    loss: 0.066941  [17000/60000]
    loss: 1.132143  [18000/60000]

    loss: 0.737702  [19000/60000]
    loss: 0.170178  [20000/60000]

    loss: 0.052959  [21000/60000]
    loss: 0.313488  [22000/60000]

    loss: 0.407618  [23000/60000]
    loss: 0.373944  [24000/60000]

    loss: 0.256424  [25000/60000]
    loss: 0.184344  [26000/60000]

    loss: 0.048323  [27000/60000]
    loss: 1.073378  [28000/60000]

    loss: 0.286181  [29000/60000]

    loss: 0.195637  [30000/60000]

    loss: 0.110032  [31000/60000]
    loss: 0.498481  [32000/60000]

    loss: 0.407290  [33000/60000]
    loss: 0.600758  [34000/60000]

    loss: 0.315671  [35000/60000]
    loss: 0.123035  [36000/60000]

    loss: 0.701893  [37000/60000]
    loss: 0.315526  [38000/60000]

    loss: 0.572403  [39000/60000]
    loss: 0.923053  [40000/60000]

    loss: 0.120233  [41000/60000]
    loss: 0.069061  [42000/60000]

    loss: 0.479645  [43000/60000]
    loss: 0.302547  [44000/60000]

    loss: 0.512247  [45000/60000]
    loss: 0.432458  [46000/60000]

    loss: 0.128307  [47000/60000]
    loss: 0.299637  [48000/60000]

    loss: 0.450630  [49000/60000]

    loss: 0.119992  [50000/60000]

    loss: 0.830663  [51000/60000]

    loss: 0.248129  [52000/60000]

    loss: 0.472861  [53000/60000]

    loss: 0.346110  [54000/60000]

    loss: 0.391488  [55000/60000]

    loss: 0.228210  [56000/60000]

    loss: 0.455863  [57000/60000]

    loss: 0.592078  [58000/60000]

    loss: 0.401325  [59000/60000]

    Test Error: 
     Accuracy: 110.6%, Avg loss: 0.373217 

    Epoch 4
    -------------------------------
    loss: 0.481119  [    0/60000]

    loss: 0.446243  [ 1000/60000]
    loss: 0.211083  [ 2000/60000]

    loss: 0.461669  [ 3000/60000]

    loss: 0.060439  [ 4000/60000]
    loss: 0.213278  [ 5000/60000]

    loss: 0.558812  [ 6000/60000]
    loss: 0.408733  [ 7000/60000]

    loss: 0.295029  [ 8000/60000]

    loss: 0.260852  [ 9000/60000]
    loss: 0.183164  [10000/60000]

    loss: 0.192481  [11000/60000]
    loss: 0.452313  [12000/60000]

    loss: 0.316820  [13000/60000]
    loss: 0.095695  [14000/60000]

    loss: 0.156351  [15000/60000]
    loss: 1.056654  [16000/60000]

    loss: 0.063700  [17000/60000]
    loss: 1.097878  [18000/60000]

    loss: 0.690527  [19000/60000]
    loss: 0.153149  [20000/60000]

    loss: 0.039941  [21000/60000]
    loss: 0.268286  [22000/60000]

    loss: 0.377895  [23000/60000]
    loss: 0.352848  [24000/60000]

    loss: 0.226390  [25000/60000]
    loss: 0.136183  [26000/60000]

    loss: 0.042487  [27000/60000]
    loss: 1.038781  [28000/60000]

    loss: 0.232577  [29000/60000]
    loss: 0.134561  [30000/60000]

    loss: 0.113128  [31000/60000]
    loss: 0.512031  [32000/60000]

    loss: 0.319918  [33000/60000]

    loss: 0.582166  [34000/60000]
    loss: 0.257864  [35000/60000]

    loss: 0.096914  [36000/60000]
    loss: 0.649854  [37000/60000]

    loss: 0.315161  [38000/60000]
    loss: 0.556753  [39000/60000]

    loss: 0.853703  [40000/60000]
    loss: 0.109040  [41000/60000]

    loss: 0.064148  [42000/60000]

    loss: 0.434653  [43000/60000]
    loss: 0.245456  [44000/60000]

    loss: 0.510700  [45000/60000]
    loss: 0.421976  [46000/60000]

    loss: 0.104753  [47000/60000]
    loss: 0.281256  [48000/60000]

    loss: 0.387987  [49000/60000]

    loss: 0.128431  [50000/60000]

    loss: 0.810879  [51000/60000]
    loss: 0.223144  [52000/60000]

    loss: 0.441438  [53000/60000]
    loss: 0.327051  [54000/60000]

    loss: 0.361899  [55000/60000]
    loss: 0.184773  [56000/60000]

    loss: 0.452278  [57000/60000]
    loss: 0.549039  [58000/60000]

    loss: 0.381790  [59000/60000]

    Test Error: 
     Accuracy: 112.9%, Avg loss: 0.349858 

    Epoch 5
    -------------------------------
    loss: 0.429544  [    0/60000]

    loss: 0.489462  [ 1000/60000]

    loss: 0.204266  [ 2000/60000]
    loss: 0.471776  [ 3000/60000]

    loss: 0.051942  [ 4000/60000]

    loss: 0.226828  [ 5000/60000]

    loss: 0.546390  [ 6000/60000]

    loss: 0.368215  [ 7000/60000]
    loss: 0.304794  [ 8000/60000]

    loss: 0.225832  [ 9000/60000]

    loss: 0.149489  [10000/60000]

    loss: 0.148379  [11000/60000]
    loss: 0.404917  [12000/60000]

    loss: 0.279988  [13000/60000]
    loss: 0.078912  [14000/60000]

    loss: 0.137655  [15000/60000]
    loss: 1.096020  [16000/60000]

    loss: 0.063260  [17000/60000]

    loss: 1.027669  [18000/60000]
    loss: 0.673573  [19000/60000]

    loss: 0.148516  [20000/60000]
    loss: 0.033067  [21000/60000]

    loss: 0.230424  [22000/60000]

    loss: 0.352551  [23000/60000]
    loss: 0.328726  [24000/60000]

    loss: 0.190954  [25000/60000]

    loss: 0.112787  [26000/60000]
    loss: 0.039296  [27000/60000]

    loss: 1.024472  [28000/60000]
    loss: 0.189638  [29000/60000]

    loss: 0.096773  [30000/60000]

    loss: 0.118896  [31000/60000]

    loss: 0.493656  [32000/60000]
    loss: 0.292365  [33000/60000]

    loss: 0.557353  [34000/60000]
    loss: 0.227794  [35000/60000]

    loss: 0.086265  [36000/60000]

    loss: 0.608628  [37000/60000]
    loss: 0.314347  [38000/60000]

    loss: 0.532585  [39000/60000]

    loss: 0.806146  [40000/60000]
    loss: 0.097430  [41000/60000]

    loss: 0.058370  [42000/60000]
    loss: 0.390279  [43000/60000]

    loss: 0.214122  [44000/60000]
    loss: 0.492132  [45000/60000]

    loss: 0.422791  [46000/60000]

    loss: 0.090093  [47000/60000]

    loss: 0.280901  [48000/60000]
    loss: 0.354855  [49000/60000]

    loss: 0.113980  [50000/60000]
    loss: 0.797323  [51000/60000]

    loss: 0.200477  [52000/60000]
    loss: 0.404075  [53000/60000]

    loss: 0.302739  [54000/60000]
    loss: 0.325606  [55000/60000]

    loss: 0.158334  [56000/60000]

    loss: 0.459957  [57000/60000]

    loss: 0.500143  [58000/60000]

    loss: 0.389526  [59000/60000]

    Test Error: 
     Accuracy: 112.0%, Avg loss: 0.328879 

    Training completed

## Comments & questions
