---
title: Training a model
description: Zoom
colordes: "#e86e0a"
slug: 14_training_nn
weight: 14
execute:
  error: true
format: hugo
jupyter: python3
---



## Packages, DataLoaders, model

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

While the parameters of a model (weights and biases) are the values that get adjusted through training, hyperparameters control the training process.

They include:

-   batch size: number of samples passed through the model before the parameters are updated,
-   number of epochs: number iterations,
-   learning rate: size of the incremental changes to model parameters at each iteration. Smaller values yield slow learning speed, while large values may miss minima.

``` python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## Define the loss function

To assess the predicted outputs of our model against the true values from the labels, we need a loss function (e.g. mean square error for regressions: `nn.MSELoss` or negative log likelihood for classification: `nn.NLLLoss`)

The machine learning literature is rich in information about various loss functions.

Here is an example with `nn.CrossEntropyLoss` which combines `nn.LogSoftmax` and `nn.NLLLoss`:

``` python
loss_fn = nn.CrossEntropyLoss()
```

## Initialize the optimizer

Optimization algorithms determine how the model parameters get adjusted at each iteration.

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
    loss: 2.290078  [    0/60000]

    loss: 2.175909  [ 1000/60000]

    loss: 1.752827  [ 2000/60000]
    loss: 1.483781  [ 3000/60000]

    loss: 1.075751  [ 4000/60000]
    loss: 1.009195  [ 5000/60000]

    loss: 1.370676  [ 6000/60000]
    loss: 0.801355  [ 7000/60000]

    loss: 0.951855  [ 8000/60000]
    loss: 0.909457  [ 9000/60000]

    loss: 0.592998  [10000/60000]
    loss: 0.855657  [11000/60000]

    loss: 1.130125  [12000/60000]
    loss: 0.675471  [13000/60000]

    loss: 0.498876  [14000/60000]
    loss: 0.644413  [15000/60000]

    loss: 1.004657  [16000/60000]
    loss: 0.331907  [17000/60000]

    loss: 1.661913  [18000/60000]
    loss: 0.743550  [19000/60000]

    loss: 0.412794  [20000/60000]
    loss: 0.191642  [21000/60000]

    loss: 0.384772  [22000/60000]
    loss: 0.771718  [23000/60000]

    loss: 0.639409  [24000/60000]
    loss: 0.482610  [25000/60000]

    loss: 0.396803  [26000/60000]
    loss: 0.149435  [27000/60000]

    loss: 1.124946  [28000/60000]
    loss: 0.647954  [29000/60000]

    loss: 0.555405  [30000/60000]
    loss: 0.181305  [31000/60000]

    loss: 0.608224  [32000/60000]
    loss: 0.706331  [33000/60000]

    loss: 0.798880  [34000/60000]
    loss: 0.466490  [35000/60000]

    loss: 0.253191  [36000/60000]
    loss: 0.792107  [37000/60000]

    loss: 0.489095  [38000/60000]
    loss: 0.537872  [39000/60000]

    loss: 0.865563  [40000/60000]
    loss: 0.345112  [41000/60000]

    loss: 0.160820  [42000/60000]
    loss: 0.697515  [43000/60000]

    loss: 0.428813  [44000/60000]
    loss: 0.698016  [45000/60000]

    loss: 0.509303  [46000/60000]
    loss: 0.281869  [47000/60000]

    loss: 0.522103  [48000/60000]
    loss: 0.607402  [49000/60000]

    loss: 0.112917  [50000/60000]
    loss: 0.950076  [51000/60000]

    loss: 0.237514  [52000/60000]
    loss: 0.693327  [53000/60000]

    loss: 0.429077  [54000/60000]
    loss: 0.479292  [55000/60000]

    loss: 0.331490  [56000/60000]
    loss: 0.679510  [57000/60000]

    loss: 0.654126  [58000/60000]
    loss: 0.445053  [59000/60000]

    Test Error: 
     Accuracy: 92.2%, Avg loss: 0.476131 

    Epoch 2
    -------------------------------
    loss: 0.674914  [    0/60000]
    loss: 0.355204  [ 1000/60000]

    loss: 0.249866  [ 2000/60000]
    loss: 0.535685  [ 3000/60000]

    loss: 0.101532  [ 4000/60000]
    loss: 0.261092  [ 5000/60000]

    loss: 0.649162  [ 6000/60000]
    loss: 0.511624  [ 7000/60000]

    loss: 0.331466  [ 8000/60000]
    loss: 0.451469  [ 9000/60000]

    loss: 0.267864  [10000/60000]
    loss: 0.408424  [11000/60000]

    loss: 0.659958  [12000/60000]
    loss: 0.446095  [13000/60000]

    loss: 0.175014  [14000/60000]
    loss: 0.275037  [15000/60000]

    loss: 0.976900  [16000/60000]
    loss: 0.086918  [17000/60000]

    loss: 1.188710  [18000/60000]
    loss: 0.779520  [19000/60000]

    loss: 0.193459  [20000/60000]
    loss: 0.062446  [21000/60000]

    loss: 0.320613  [22000/60000]
    loss: 0.442150  [23000/60000]

    loss: 0.419405  [24000/60000]
    loss: 0.313024  [25000/60000]

    loss: 0.227275  [26000/60000]
    loss: 0.062590  [27000/60000]

    loss: 1.103735  [28000/60000]
    loss: 0.399015  [29000/60000]

    loss: 0.301195  [30000/60000]
    loss: 0.110073  [31000/60000]

    loss: 0.471572  [32000/60000]
    loss: 0.538054  [33000/60000]

    loss: 0.659879  [34000/60000]
    loss: 0.408837  [35000/60000]

    loss: 0.155935  [36000/60000]
    loss: 0.706468  [37000/60000]

    loss: 0.358466  [38000/60000]
    loss: 0.559195  [39000/60000]

    loss: 0.880589  [40000/60000]
    loss: 0.159385  [41000/60000]

    loss: 0.081394  [42000/60000]
    loss: 0.549725  [43000/60000]

    loss: 0.374594  [44000/60000]
    loss: 0.579694  [45000/60000]

    loss: 0.459086  [46000/60000]
    loss: 0.157694  [47000/60000]

    loss: 0.363343  [48000/60000]
    loss: 0.519822  [49000/60000]

    loss: 0.118100  [50000/60000]
    loss: 0.849384  [51000/60000]

    loss: 0.221252  [52000/60000]
    loss: 0.515981  [53000/60000]

    loss: 0.347451  [54000/60000]
    loss: 0.427203  [55000/60000]

    loss: 0.271499  [56000/60000]
    loss: 0.527861  [57000/60000]

    loss: 0.617475  [58000/60000]
    loss: 0.394153  [59000/60000]

    Test Error: 
     Accuracy: 95.8%, Avg loss: 0.410446 

    Epoch 3
    -------------------------------
    loss: 0.565882  [    0/60000]
    loss: 0.374095  [ 1000/60000]

    loss: 0.203961  [ 2000/60000]
    loss: 0.493314  [ 3000/60000]

    loss: 0.064018  [ 4000/60000]
    loss: 0.207118  [ 5000/60000]

    loss: 0.536460  [ 6000/60000]
    loss: 0.439078  [ 7000/60000]

    loss: 0.305634  [ 8000/60000]
    loss: 0.312661  [ 9000/60000]

    loss: 0.217111  [10000/60000]
    loss: 0.274358  [11000/60000]

    loss: 0.576091  [12000/60000]
    loss: 0.367616  [13000/60000]

    loss: 0.138128  [14000/60000]
    loss: 0.193294  [15000/60000]

    loss: 1.027959  [16000/60000]
    loss: 0.066590  [17000/60000]

    loss: 1.148113  [18000/60000]
    loss: 0.741657  [19000/60000]

    loss: 0.165835  [20000/60000]
    loss: 0.046698  [21000/60000]

    loss: 0.302461  [22000/60000]
    loss: 0.380731  [23000/60000]

    loss: 0.377939  [24000/60000]
    loss: 0.286783  [25000/60000]

    loss: 0.162386  [26000/60000]
    loss: 0.044855  [27000/60000]

    loss: 1.089164  [28000/60000]
    loss: 0.301782  [29000/60000]

    loss: 0.198920  [30000/60000]
    loss: 0.104264  [31000/60000]

    loss: 0.484383  [32000/60000]
    loss: 0.405189  [33000/60000]

    loss: 0.585919  [34000/60000]
    loss: 0.388771  [35000/60000]

    loss: 0.105924  [36000/60000]
    loss: 0.667646  [37000/60000]

    loss: 0.324323  [38000/60000]
    loss: 0.626489  [39000/60000]

    loss: 0.868886  [40000/60000]
    loss: 0.115498  [41000/60000]

    loss: 0.064228  [42000/60000]
    loss: 0.473193  [43000/60000]

    loss: 0.316293  [44000/60000]
    loss: 0.553922  [45000/60000]

    loss: 0.426380  [46000/60000]
    loss: 0.131278  [47000/60000]

    loss: 0.307736  [48000/60000]
    loss: 0.464401  [49000/60000]

    loss: 0.118437  [50000/60000]
    loss: 0.765825  [51000/60000]

    loss: 0.215936  [52000/60000]
    loss: 0.434501  [53000/60000]

    loss: 0.308184  [54000/60000]
    loss: 0.413248  [55000/60000]

    loss: 0.222270  [56000/60000]
    loss: 0.476505  [57000/60000]

    loss: 0.628386  [58000/60000]
    loss: 0.383033  [59000/60000]

    Test Error: 
     Accuracy: 97.1%, Avg loss: 0.374140 

    Epoch 4
    -------------------------------
    loss: 0.509754  [    0/60000]
    loss: 0.429466  [ 1000/60000]

    loss: 0.196705  [ 2000/60000]
    loss: 0.462557  [ 3000/60000]

    loss: 0.048749  [ 4000/60000]
    loss: 0.194309  [ 5000/60000]

    loss: 0.514447  [ 6000/60000]
    loss: 0.384651  [ 7000/60000]

    loss: 0.314780  [ 8000/60000]
    loss: 0.234213  [ 9000/60000]

    loss: 0.191456  [10000/60000]
    loss: 0.190509  [11000/60000]

    loss: 0.498667  [12000/60000]
    loss: 0.314666  [13000/60000]

    loss: 0.111425  [14000/60000]
    loss: 0.154640  [15000/60000]

    loss: 1.113726  [16000/60000]
    loss: 0.060348  [17000/60000]

    loss: 1.123035  [18000/60000]
    loss: 0.697651  [19000/60000]

    loss: 0.145172  [20000/60000]
    loss: 0.042126  [21000/60000]

    loss: 0.249148  [22000/60000]
    loss: 0.350588  [23000/60000]

    loss: 0.357840  [24000/60000]
    loss: 0.264470  [25000/60000]

    loss: 0.123823  [26000/60000]
    loss: 0.038974  [27000/60000]

    loss: 1.077199  [28000/60000]
    loss: 0.249021  [29000/60000]

    loss: 0.142029  [30000/60000]
    loss: 0.105350  [31000/60000]

    loss: 0.472939  [32000/60000]
    loss: 0.332272  [33000/60000]

    loss: 0.549120  [34000/60000]
    loss: 0.362499  [35000/60000]

    loss: 0.090557  [36000/60000]
    loss: 0.595271  [37000/60000]

    loss: 0.321202  [38000/60000]
    loss: 0.607974  [39000/60000]

    loss: 0.799913  [40000/60000]
    loss: 0.095401  [41000/60000]

    loss: 0.057795  [42000/60000]
    loss: 0.433303  [43000/60000]

    loss: 0.256051  [44000/60000]
    loss: 0.541217  [45000/60000]

    loss: 0.423699  [46000/60000]
    loss: 0.123898  [47000/60000]

    loss: 0.269432  [48000/60000]
    loss: 0.411791  [49000/60000]

    loss: 0.126552  [50000/60000]
    loss: 0.744011  [51000/60000]

    loss: 0.213943  [52000/60000]
    loss: 0.387209  [53000/60000]

    loss: 0.292156  [54000/60000]
    loss: 0.387254  [55000/60000]

    loss: 0.184812  [56000/60000]
    loss: 0.441932  [57000/60000]

    loss: 0.590237  [58000/60000]
    loss: 0.375701  [59000/60000]

    Test Error: 
     Accuracy: 99.8%, Avg loss: 0.350540 

    Epoch 5
    -------------------------------
    loss: 0.463481  [    0/60000]
    loss: 0.461113  [ 1000/60000]

    loss: 0.182303  [ 2000/60000]
    loss: 0.458964  [ 3000/60000]

    loss: 0.043798  [ 4000/60000]
    loss: 0.188478  [ 5000/60000]

    loss: 0.509040  [ 6000/60000]
    loss: 0.348380  [ 7000/60000]

    loss: 0.327356  [ 8000/60000]
    loss: 0.188386  [ 9000/60000]

    loss: 0.156696  [10000/60000]
    loss: 0.154098  [11000/60000]

    loss: 0.443449  [12000/60000]
    loss: 0.289711  [13000/60000]

    loss: 0.088884  [14000/60000]
    loss: 0.127516  [15000/60000]

    loss: 1.164954  [16000/60000]
    loss: 0.061881  [17000/60000]

    loss: 1.077961  [18000/60000]
    loss: 0.647685  [19000/60000]

    loss: 0.134136  [20000/60000]
    loss: 0.038252  [21000/60000]

    loss: 0.217067  [22000/60000]
    loss: 0.315540  [23000/60000]

    loss: 0.344841  [24000/60000]
    loss: 0.251260  [25000/60000]

    loss: 0.109184  [26000/60000]
    loss: 0.033465  [27000/60000]

    loss: 1.052413  [28000/60000]
    loss: 0.209121  [29000/60000]

    loss: 0.116288  [30000/60000]
    loss: 0.100228  [31000/60000]

    loss: 0.465138  [32000/60000]
    loss: 0.303976  [33000/60000]

    loss: 0.534196  [34000/60000]
    loss: 0.334181  [35000/60000]

    loss: 0.080613  [36000/60000]
    loss: 0.579001  [37000/60000]

    loss: 0.321934  [38000/60000]
    loss: 0.542591  [39000/60000]

    loss: 0.733868  [40000/60000]
    loss: 0.091428  [41000/60000]

    loss: 0.056304  [42000/60000]
    loss: 0.400804  [43000/60000]

    loss: 0.213983  [44000/60000]
    loss: 0.550695  [45000/60000]

    loss: 0.417832  [46000/60000]
    loss: 0.112926  [47000/60000]

    loss: 0.254434  [48000/60000]
    loss: 0.382358  [49000/60000]

    loss: 0.111378  [50000/60000]
    loss: 0.713137  [51000/60000]

    loss: 0.225667  [52000/60000]
    loss: 0.342614  [53000/60000]

    loss: 0.270426  [54000/60000]
    loss: 0.364463  [55000/60000]

    loss: 0.149493  [56000/60000]
    loss: 0.429284  [57000/60000]

    loss: 0.568922  [58000/60000]
    loss: 0.387797  [59000/60000]

    Test Error: 
     Accuracy: 100.0%, Avg loss: 0.332800 

    Training completed

## Comments & questions
