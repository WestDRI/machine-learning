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



<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js" integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg==" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
<script src="https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js" crossorigin="anonymous"></script>


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
    root="/project/def-sponsor00/data/",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root="/project/def-sponsor00/data/",
    train=False,
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

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./FashionMNIST/raw/train-images-idx3-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"bf434dc1174c439093a67154792d0be0","version_major":2,"version_minor":0}
</script>

    Extracting ./FashionMNIST/raw/train-images-idx3-ubyte.gz to ./FashionMNIST/raw


    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./FashionMNIST/raw/train-labels-idx1-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"996c8071208a4715afdc1854e0ccd3b6","version_major":2,"version_minor":0}
</script>

    Extracting ./FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./FashionMNIST/raw

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./FashionMNIST/raw/t10k-images-idx3-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"fbabdd07e31946d88e6a4d8b476691ed","version_major":2,"version_minor":0}
</script>

    Extracting ./FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./FashionMNIST/raw

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"9796514d1fd64d93a3ccc0fbd761688b","version_major":2,"version_minor":0}
</script>

    Extracting ./FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./FashionMNIST/raw

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
    loss: 2.313894  [    0/60000]

    loss: 2.209583  [ 1000/60000]

    loss: 1.825431  [ 2000/60000]

    loss: 1.566175  [ 3000/60000]

    loss: 1.108923  [ 4000/60000]

    loss: 1.006603  [ 5000/60000]
    loss: 1.413068  [ 6000/60000]

    loss: 0.803126  [ 7000/60000]
    loss: 0.939774  [ 8000/60000]

    loss: 0.927773  [ 9000/60000]
    loss: 0.612259  [10000/60000]

    loss: 0.840623  [11000/60000]
    loss: 1.120255  [12000/60000]

    loss: 0.665815  [13000/60000]
    loss: 0.487792  [14000/60000]

    loss: 0.642110  [15000/60000]
    loss: 1.016150  [16000/60000]

    loss: 0.336110  [17000/60000]
    loss: 1.556458  [18000/60000]

    loss: 0.755764  [19000/60000]
    loss: 0.407435  [20000/60000]

    loss: 0.194701  [21000/60000]
    loss: 0.361907  [22000/60000]

    loss: 0.742247  [23000/60000]
    loss: 0.672527  [24000/60000]

    loss: 0.466937  [25000/60000]
    loss: 0.422173  [26000/60000]

    loss: 0.134426  [27000/60000]
    loss: 1.093594  [28000/60000]

    loss: 0.653817  [29000/60000]
    loss: 0.547656  [30000/60000]

    loss: 0.174427  [31000/60000]
    loss: 0.605598  [32000/60000]

    loss: 0.714456  [33000/60000]
    loss: 0.813955  [34000/60000]

    loss: 0.481885  [35000/60000]
    loss: 0.239479  [36000/60000]

    loss: 0.802305  [37000/60000]
    loss: 0.471210  [38000/60000]

    loss: 0.555649  [39000/60000]
    loss: 0.914729  [40000/60000]

    loss: 0.342048  [41000/60000]
    loss: 0.157393  [42000/60000]

    loss: 0.684286  [43000/60000]
    loss: 0.417549  [44000/60000]

    loss: 0.691615  [45000/60000]
    loss: 0.505846  [46000/60000]

    loss: 0.314346  [47000/60000]
    loss: 0.505909  [48000/60000]

    loss: 0.593913  [49000/60000]
    loss: 0.118894  [50000/60000]

    loss: 1.069959  [51000/60000]
    loss: 0.235975  [52000/60000]

    loss: 0.694713  [53000/60000]
    loss: 0.435227  [54000/60000]

    loss: 0.477341  [55000/60000]
    loss: 0.373612  [56000/60000]

    loss: 0.676326  [57000/60000]
    loss: 0.661223  [58000/60000]

    loss: 0.456003  [59000/60000]

    Test Error: 
     Accuracy: 103.7%, Avg loss: 0.472968 

    Epoch 2
    -------------------------------
    loss: 0.701486  [    0/60000]
    loss: 0.378306  [ 1000/60000]

    loss: 0.267802  [ 2000/60000]
    loss: 0.522529  [ 3000/60000]

    loss: 0.091644  [ 4000/60000]
    loss: 0.266027  [ 5000/60000]

    loss: 0.678095  [ 6000/60000]
    loss: 0.523694  [ 7000/60000]

    loss: 0.341322  [ 8000/60000]
    loss: 0.487370  [ 9000/60000]

    loss: 0.287361  [10000/60000]
    loss: 0.401389  [11000/60000]

    loss: 0.650608  [12000/60000]
    loss: 0.445530  [13000/60000]

    loss: 0.164422  [14000/60000]
    loss: 0.246646  [15000/60000]

    loss: 1.016836  [16000/60000]

    loss: 0.083177  [17000/60000]

    loss: 1.129416  [18000/60000]
    loss: 0.730492  [19000/60000]

    loss: 0.207064  [20000/60000]
    loss: 0.074803  [21000/60000]

    loss: 0.301070  [22000/60000]
    loss: 0.427675  [23000/60000]

    loss: 0.450497  [24000/60000]
    loss: 0.320961  [25000/60000]

    loss: 0.256764  [26000/60000]

    loss: 0.061158  [27000/60000]
    loss: 1.112471  [28000/60000]

    loss: 0.399782  [29000/60000]
    loss: 0.304485  [30000/60000]

    loss: 0.107120  [31000/60000]

    loss: 0.479984  [32000/60000]

    loss: 0.529688  [33000/60000]
    loss: 0.690905  [34000/60000]

    loss: 0.430148  [35000/60000]
    loss: 0.153147  [36000/60000]

    loss: 0.736496  [37000/60000]
    loss: 0.341141  [38000/60000]

    loss: 0.612477  [39000/60000]

    loss: 0.929790  [40000/60000]
    loss: 0.171491  [41000/60000]

    loss: 0.081986  [42000/60000]
    loss: 0.523186  [43000/60000]

    loss: 0.367308  [44000/60000]
    loss: 0.574111  [45000/60000]

    loss: 0.467557  [46000/60000]
    loss: 0.183672  [47000/60000]

    loss: 0.360408  [48000/60000]

    loss: 0.526477  [49000/60000]

    loss: 0.127008  [50000/60000]
    loss: 0.965567  [51000/60000]

    loss: 0.214584  [52000/60000]

    loss: 0.527377  [53000/60000]

    loss: 0.370190  [54000/60000]
    loss: 0.408935  [55000/60000]

    loss: 0.295107  [56000/60000]
    loss: 0.530069  [57000/60000]

    loss: 0.634487  [58000/60000]
    loss: 0.408090  [59000/60000]

    Test Error: 
     Accuracy: 108.6%, Avg loss: 0.408476 

    Epoch 3
    -------------------------------
    loss: 0.589625  [    0/60000]
    loss: 0.390567  [ 1000/60000]

    loss: 0.237769  [ 2000/60000]

    loss: 0.460030  [ 3000/60000]

    loss: 0.056316  [ 4000/60000]

    loss: 0.211697  [ 5000/60000]

    loss: 0.558873  [ 6000/60000]

    loss: 0.466138  [ 7000/60000]

    loss: 0.293455  [ 8000/60000]

    loss: 0.337856  [ 9000/60000]

    loss: 0.227861  [10000/60000]

    loss: 0.264908  [11000/60000]

    loss: 0.545475  [12000/60000]

    loss: 0.356967  [13000/60000]

    loss: 0.127376  [14000/60000]

    loss: 0.172921  [15000/60000]

    loss: 1.090881  [16000/60000]

    loss: 0.064057  [17000/60000]

    loss: 1.086788  [18000/60000]

    loss: 0.698609  [19000/60000]

    loss: 0.166774  [20000/60000]

    loss: 0.058505  [21000/60000]

    loss: 0.277547  [22000/60000]

    loss: 0.340083  [23000/60000]

    loss: 0.417019  [24000/60000]

    loss: 0.302526  [25000/60000]

    loss: 0.182446  [26000/60000]

    loss: 0.044894  [27000/60000]

    loss: 1.089933  [28000/60000]

    loss: 0.298356  [29000/60000]

    loss: 0.200580  [30000/60000]

    loss: 0.098322  [31000/60000]

    loss: 0.485597  [32000/60000]

    loss: 0.407959  [33000/60000]

    loss: 0.620739  [34000/60000]

    loss: 0.382905  [35000/60000]

    loss: 0.107402  [36000/60000]

    loss: 0.684698  [37000/60000]

    loss: 0.307851  [38000/60000]

    loss: 0.621548  [39000/60000]

    loss: 0.875995  [40000/60000]

    loss: 0.121690  [41000/60000]

    loss: 0.069163  [42000/60000]

    loss: 0.452378  [43000/60000]

    loss: 0.306732  [44000/60000]

    loss: 0.534353  [45000/60000]

    loss: 0.439734  [46000/60000]

    loss: 0.145556  [47000/60000]

    loss: 0.297056  [48000/60000]

    loss: 0.453122  [49000/60000]

    loss: 0.129692  [50000/60000]

    loss: 0.890579  [51000/60000]

    loss: 0.208035  [52000/60000]

    loss: 0.450671  [53000/60000]

    loss: 0.342771  [54000/60000]

    loss: 0.405848  [55000/60000]

    loss: 0.245603  [56000/60000]

    loss: 0.489146  [57000/60000]
    loss: 0.605939  [58000/60000]

    loss: 0.398694  [59000/60000]

    Test Error: 
     Accuracy: 110.2%, Avg loss: 0.372263 

    Epoch 4
    -------------------------------
    loss: 0.532061  [    0/60000]
    loss: 0.443890  [ 1000/60000]

    loss: 0.226630  [ 2000/60000]

    loss: 0.431535  [ 3000/60000]

    loss: 0.046243  [ 4000/60000]

    loss: 0.212723  [ 5000/60000]
    loss: 0.533458  [ 6000/60000]

    loss: 0.409180  [ 7000/60000]
    loss: 0.292523  [ 8000/60000]

    loss: 0.249466  [ 9000/60000]
    loss: 0.185609  [10000/60000]

    loss: 0.189724  [11000/60000]
    loss: 0.470238  [12000/60000]

    loss: 0.325083  [13000/60000]

    loss: 0.098501  [14000/60000]
    loss: 0.143514  [15000/60000]

    loss: 1.145604  [16000/60000]

    loss: 0.060229  [17000/60000]
    loss: 1.041621  [18000/60000]

    loss: 0.645271  [19000/60000]

    loss: 0.152798  [20000/60000]

    loss: 0.051467  [21000/60000]
    loss: 0.231872  [22000/60000]

    loss: 0.308140  [23000/60000]
    loss: 0.386025  [24000/60000]

    loss: 0.262250  [25000/60000]

    loss: 0.140432  [26000/60000]
    loss: 0.038993  [27000/60000]

    loss: 1.104994  [28000/60000]

    loss: 0.250279  [29000/60000]

    loss: 0.146712  [30000/60000]

    loss: 0.094324  [31000/60000]

    loss: 0.489743  [32000/60000]
    loss: 0.331888  [33000/60000]

    loss: 0.587607  [34000/60000]
    loss: 0.345963  [35000/60000]

    loss: 0.086927  [36000/60000]
    loss: 0.631197  [37000/60000]

    loss: 0.300410  [38000/60000]
    loss: 0.604595  [39000/60000]

    loss: 0.804568  [40000/60000]

    loss: 0.113696  [41000/60000]

    loss: 0.058610  [42000/60000]

    loss: 0.427789  [43000/60000]

    loss: 0.252483  [44000/60000]

    loss: 0.517076  [45000/60000]

    loss: 0.408849  [46000/60000]

    loss: 0.132180  [47000/60000]

    loss: 0.258762  [48000/60000]

    loss: 0.385793  [49000/60000]

    loss: 0.117567  [50000/60000]
    loss: 0.848096  [51000/60000]

    loss: 0.212472  [52000/60000]

    loss: 0.400717  [53000/60000]
    loss: 0.334782  [54000/60000]

    loss: 0.391648  [55000/60000]
    loss: 0.210674  [56000/60000]

    loss: 0.489246  [57000/60000]
    loss: 0.575880  [58000/60000]

    loss: 0.396307  [59000/60000]

    Test Error: 
     Accuracy: 111.3%, Avg loss: 0.349180 

    Epoch 5
    -------------------------------
    loss: 0.483897  [    0/60000]

    loss: 0.482102  [ 1000/60000]
    loss: 0.223865  [ 2000/60000]

    loss: 0.433387  [ 3000/60000]
    loss: 0.039494  [ 4000/60000]

    loss: 0.219176  [ 5000/60000]
    loss: 0.513556  [ 6000/60000]

    loss: 0.376667  [ 7000/60000]
    loss: 0.305540  [ 8000/60000]

    loss: 0.212830  [ 9000/60000]
    loss: 0.146233  [10000/60000]

    loss: 0.142688  [11000/60000]
    loss: 0.426622  [12000/60000]

    loss: 0.298005  [13000/60000]
    loss: 0.080610  [14000/60000]

    loss: 0.113162  [15000/60000]
    loss: 1.187650  [16000/60000]

    loss: 0.063121  [17000/60000]
    loss: 0.994065  [18000/60000]

    loss: 0.584324  [19000/60000]
    loss: 0.149704  [20000/60000]

    loss: 0.047375  [21000/60000]
    loss: 0.200011  [22000/60000]

    loss: 0.290627  [23000/60000]
    loss: 0.363045  [24000/60000]

    loss: 0.230466  [25000/60000]
    loss: 0.113239  [26000/60000]

    loss: 0.036227  [27000/60000]
    loss: 1.114385  [28000/60000]

    loss: 0.212635  [29000/60000]
    loss: 0.107470  [30000/60000]

    loss: 0.094590  [31000/60000]
    loss: 0.481113  [32000/60000]

    loss: 0.282456  [33000/60000]
    loss: 0.569291  [34000/60000]

    loss: 0.300268  [35000/60000]

    loss: 0.075891  [36000/60000]

    loss: 0.571429  [37000/60000]

    loss: 0.309798  [38000/60000]

    loss: 0.578555  [39000/60000]

    loss: 0.753653  [40000/60000]
    loss: 0.106412  [41000/60000]

    loss: 0.056287  [42000/60000]
    loss: 0.409297  [43000/60000]

    loss: 0.198123  [44000/60000]
    loss: 0.504752  [45000/60000]

    loss: 0.424229  [46000/60000]
    loss: 0.117185  [47000/60000]

    loss: 0.243067  [48000/60000]
    loss: 0.344168  [49000/60000]

    loss: 0.109870  [50000/60000]
    loss: 0.798450  [51000/60000]

    loss: 0.196153  [52000/60000]

    loss: 0.379001  [53000/60000]

    loss: 0.313801  [54000/60000]
    loss: 0.367338  [55000/60000]

    loss: 0.158342  [56000/60000]
    loss: 0.491715  [57000/60000]

    loss: 0.552428  [58000/60000]
    loss: 0.378269  [59000/60000]

    Test Error: 
     Accuracy: 112.7%, Avg loss: 0.333500 

    Training completed

## Comments & questions

<script type=application/vnd.jupyter.widget-state+json>
{"state":{"0044fc29835247f1a69011a983777be3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_a9e76ce61bf94cd1ba3c378a88f99d0e","placeholder":"​","style":"IPY_MODEL_36cabf2e3c914178ae0556520f3d82d1","value":" 5148/5148 [00:00&lt;00:00, 290476.46it/s]"}},"01b808342b094e1eb149115611901f00":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_2b5ff5eb58804034828781121466a512","placeholder":"​","style":"IPY_MODEL_83b11e97b0f54fdabbe7d48b6ca78524","value":" 29515/29515 [00:00&lt;00:00, 143682.39it/s]"}},"04703999fff742339c5ea060723b37b0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"0e31aaa0090144bc8ff918bd4f28738a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"1456012b77af45bcb7e10903a13352ef":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_7c627309bc9649e1b44ea5f2ffb41bd0","placeholder":"​","style":"IPY_MODEL_78c6aed8949c44bd9c11fcf9e53fa59e","value":"100%"}},"1655f9a3c9ea436680484b8b8c921744":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_beacc9cb3ca648988151b4fe0ed091e2","max":29515,"min":0,"orientation":"horizontal","style":"IPY_MODEL_79cb6e6ec78c4888b7c49dcbf227d451","value":29515}},"1f244d4792b7468c9cc9311e498cbb5d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_5b92e14037ff4327a570db1048cea55e","placeholder":"​","style":"IPY_MODEL_ea389082d9ec415d9622912593077f43","value":"100%"}},"206cc818ca8f404bb163459a058d42c7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"2b5ff5eb58804034828781121466a512":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"2f7ac6316c5c411c9e3c4c58bb7b7ece":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"344064a8ba4f4656a3de7dd38751ee85":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"36cabf2e3c914178ae0556520f3d82d1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"43a6584b6b1b44d692816f0a836021fd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_75d362ac06814d24a29080e38efac899","max":5148,"min":0,"orientation":"horizontal","style":"IPY_MODEL_2f7ac6316c5c411c9e3c4c58bb7b7ece","value":5148}},"48908056fb704a2fb017d67c58c7e808":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"53379222cbac4ad1938050f54079c1a5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_206cc818ca8f404bb163459a058d42c7","placeholder":"​","style":"IPY_MODEL_344064a8ba4f4656a3de7dd38751ee85","value":" 4422102/4422102 [00:01&lt;00:00, 3579950.59it/s]"}},"5b92e14037ff4327a570db1048cea55e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"620e467d201840e58ecb6a0150d318a3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"6fba9daa9ada4e0cb8fd1cab86928f98":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"75d362ac06814d24a29080e38efac899":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"767b3ccfb0f846559c389b47c650e325":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"777cbc8eb2ec4e70ba1b0f45bc399cea":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"78c6aed8949c44bd9c11fcf9e53fa59e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"79cb6e6ec78c4888b7c49dcbf227d451":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"7c627309bc9649e1b44ea5f2ffb41bd0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"83b11e97b0f54fdabbe7d48b6ca78524":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"84be93ed27bb46b29e61668ca5293803":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"9796514d1fd64d93a3ccc0fbd761688b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_9d5c97656e5e45e891533a5b524f82c9","IPY_MODEL_43a6584b6b1b44d692816f0a836021fd","IPY_MODEL_0044fc29835247f1a69011a983777be3"],"layout":"IPY_MODEL_d9560a61aa494f3db623c8285e4afaaf"}},"996c8071208a4715afdc1854e0ccd3b6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_1456012b77af45bcb7e10903a13352ef","IPY_MODEL_1655f9a3c9ea436680484b8b8c921744","IPY_MODEL_01b808342b094e1eb149115611901f00"],"layout":"IPY_MODEL_6fba9daa9ada4e0cb8fd1cab86928f98"}},"9d5c97656e5e45e891533a5b524f82c9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_e96040a44a0a4d9186c02d2f2d75eec5","placeholder":"​","style":"IPY_MODEL_bc84a6b067b84e099807f970d6909454","value":"100%"}},"a9e76ce61bf94cd1ba3c378a88f99d0e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"bc84a6b067b84e099807f970d6909454":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"beacc9cb3ca648988151b4fe0ed091e2":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"bf434dc1174c439093a67154792d0be0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_c65e668feae9405f804db9916df78d7d","IPY_MODEL_dc7441dae88c484d833f0e76bbd296b5","IPY_MODEL_f055ee2fd2e04fb691891e7daf67e19c"],"layout":"IPY_MODEL_04703999fff742339c5ea060723b37b0"}},"c0edf81528f7416b96927415b82a7c5e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_48908056fb704a2fb017d67c58c7e808","max":4422102,"min":0,"orientation":"horizontal","style":"IPY_MODEL_dcd9f7c1b04e4bc2898523c8f746117d","value":4422102}},"c65e668feae9405f804db9916df78d7d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_84be93ed27bb46b29e61668ca5293803","placeholder":"​","style":"IPY_MODEL_0e31aaa0090144bc8ff918bd4f28738a","value":"100%"}},"d9560a61aa494f3db623c8285e4afaaf":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"dc7441dae88c484d833f0e76bbd296b5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_ea760e087e98417da72cdd122b1f1d5e","max":26421880,"min":0,"orientation":"horizontal","style":"IPY_MODEL_767b3ccfb0f846559c389b47c650e325","value":26421880}},"dcd9f7c1b04e4bc2898523c8f746117d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"dd70ad38152740f084e72b70b64757e7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"e96040a44a0a4d9186c02d2f2d75eec5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"ea389082d9ec415d9622912593077f43":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"ea760e087e98417da72cdd122b1f1d5e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"f055ee2fd2e04fb691891e7daf67e19c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_dd70ad38152740f084e72b70b64757e7","placeholder":"​","style":"IPY_MODEL_777cbc8eb2ec4e70ba1b0f45bc399cea","value":" 26421880/26421880 [00:10&lt;00:00, 6612842.60it/s]"}},"fbabdd07e31946d88e6a4d8b476691ed":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_1f244d4792b7468c9cc9311e498cbb5d","IPY_MODEL_c0edf81528f7416b96927415b82a7c5e","IPY_MODEL_53379222cbac4ad1938050f54079c1a5"],"layout":"IPY_MODEL_620e467d201840e58ecb6a0150d318a3"}}},"version_major":2,"version_minor":0}
</script>
