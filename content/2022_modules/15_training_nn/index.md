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



<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js" integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg==" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
<script src="https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js" crossorigin="anonymous"></script>


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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/train-images-idx3-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"399c2f0147734878975d276f38534341","version_major":2,"version_minor":0}
</script>

    Extracting /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/train-images-idx3-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw


    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"a1500f688261493d9ae7206b2aa8f373","version_major":2,"version_minor":0}
</script>

    Extracting /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"b13b131a542146dcba3f45fa1f258bdd","version_major":2,"version_minor":0}
</script>

    Extracting /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"19131e2b34914cdca68f929d7d007faa","version_major":2,"version_minor":0}
</script>

    Extracting /home/marie/projects/def-sponsor00/data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to /home/marie/projects/def-sponsor00/data/FashionMNIST/raw

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
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

    NameError: name 'optim' is not defined

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

    NameError: name 'optimizer' is not defined

## Comments & questions

<script type=application/vnd.jupyter.widget-state+json>
{"state":{"0206ce9b728b4a758f99417a8810961d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"0a45ff6957504a0880ce04d23fecadbd":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"121a2a1cc0824bacbe411ea9987834b6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"13616cf3207342549da3ae34888cdaf7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"178ab9312a84417bb5b53dc1b3aecc25":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"19131e2b34914cdca68f929d7d007faa":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_7923cfc21b3446e294dc9fb458a4eb5a","IPY_MODEL_abbca4650df94576a8d0d83aecce8fbd","IPY_MODEL_f980fa95b3b749b0b63937fe3517d7c3"],"layout":"IPY_MODEL_d5d643fe7714477780edd4506dcdf5a0"}},"257be1bcb2f14981bdadb4722c0dced6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"26f9e73872854b4aad4f726fafdb3051":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"29345fd652f74af596c2717552c7d9a1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"31e2a5bc6825403689ed82312c255cf5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"399c2f0147734878975d276f38534341":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_d574168d6873424daa4ce9e38f5873da","IPY_MODEL_6bce0769bf514a749ef7b438b2feed52","IPY_MODEL_b68626a37ed048bc83c71d0a9728c4ab"],"layout":"IPY_MODEL_26f9e73872854b4aad4f726fafdb3051"}},"3f7736c2d0684174ad34eed711099121":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"4312b28a1b64455284e25e172a91acf0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_bc77be48637f49c7afe85a6b85bcccd3","max":4422102,"min":0,"orientation":"horizontal","style":"IPY_MODEL_0206ce9b728b4a758f99417a8810961d","value":4422102}},"498aa9155e6844c5b5111a12fc5ca304":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"5b6af4a0a15e46a7a3e7a13a8f5e6735":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"638244282c404b1a99e82c9bdabaa68e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"662ae3bcd14242ad967dfd1fc05b1164":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_c1ca46984904465998e713abe1559fbe","placeholder":"​","style":"IPY_MODEL_121a2a1cc0824bacbe411ea9987834b6","value":"100%"}},"6bce0769bf514a749ef7b438b2feed52":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_b1423791013942b4be6c521ada19d35d","max":26421880,"min":0,"orientation":"horizontal","style":"IPY_MODEL_6d53241f952942a082f0bd3c39d4411c","value":26421880}},"6d53241f952942a082f0bd3c39d4411c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"7923cfc21b3446e294dc9fb458a4eb5a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_7f82c53523b04aa7a963b69e206c8464","placeholder":"​","style":"IPY_MODEL_31e2a5bc6825403689ed82312c255cf5","value":"100%"}},"7def5929d5dd40a08b5453431493b8c0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_e8417366f28a42729ad4b40c09f8eed7","placeholder":"​","style":"IPY_MODEL_638244282c404b1a99e82c9bdabaa68e","value":"100%"}},"7f82c53523b04aa7a963b69e206c8464":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"857550cc571d4823a34605aad36894bb":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"a1500f688261493d9ae7206b2aa8f373":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_7def5929d5dd40a08b5453431493b8c0","IPY_MODEL_ec7f72ad32424a509d02d38e32cc6a67","IPY_MODEL_dbad84c6ec704fe48eab1cd7c842ff34"],"layout":"IPY_MODEL_13616cf3207342549da3ae34888cdaf7"}},"abbca4650df94576a8d0d83aecce8fbd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_5b6af4a0a15e46a7a3e7a13a8f5e6735","max":5148,"min":0,"orientation":"horizontal","style":"IPY_MODEL_f21279904eaf45b494390ffc08265eb3","value":5148}},"ac5d63e56183416086774908f48454ac":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"b13b131a542146dcba3f45fa1f258bdd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_662ae3bcd14242ad967dfd1fc05b1164","IPY_MODEL_4312b28a1b64455284e25e172a91acf0","IPY_MODEL_e79f6c1d1f244f7e877837ad0ff527ab"],"layout":"IPY_MODEL_857550cc571d4823a34605aad36894bb"}},"b1423791013942b4be6c521ada19d35d":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"b42234d61efa400b9a53aaed884b1248":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"b68626a37ed048bc83c71d0a9728c4ab":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_ac5d63e56183416086774908f48454ac","placeholder":"​","style":"IPY_MODEL_498aa9155e6844c5b5111a12fc5ca304","value":" 26421880/26421880 [00:10&lt;00:00, 5851658.41it/s]"}},"bc77be48637f49c7afe85a6b85bcccd3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"be4feac5599a487b9ca01efdf61d29b8":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"be9d2c16016b441abecc5d616b54d99e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"DescriptionStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","description_width":""}},"c1ca46984904465998e713abe1559fbe":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"d574168d6873424daa4ce9e38f5873da":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_be4feac5599a487b9ca01efdf61d29b8","placeholder":"​","style":"IPY_MODEL_29345fd652f74af596c2717552c7d9a1","value":"100%"}},"d5d643fe7714477780edd4506dcdf5a0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"dbad84c6ec704fe48eab1cd7c842ff34":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_178ab9312a84417bb5b53dc1b3aecc25","placeholder":"​","style":"IPY_MODEL_257be1bcb2f14981bdadb4722c0dced6","value":" 29515/29515 [00:00&lt;00:00, 73100.11it/s]"}},"e79f6c1d1f244f7e877837ad0ff527ab":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_fcb5b69306804aa38a4dd6892c2b460b","placeholder":"​","style":"IPY_MODEL_b42234d61efa400b9a53aaed884b1248","value":" 4422102/4422102 [00:04&lt;00:00, 1468190.52it/s]"}},"e8417366f28a42729ad4b40c09f8eed7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"ec7f72ad32424a509d02d38e32cc6a67":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"ProgressView","bar_style":"success","description":"","description_tooltip":null,"layout":"IPY_MODEL_0a45ff6957504a0880ce04d23fecadbd","max":29515,"min":0,"orientation":"horizontal","style":"IPY_MODEL_3f7736c2d0684174ad34eed711099121","value":29515}},"f21279904eaf45b494390ffc08265eb3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"f3cd3445a973409083431a24e56a5b4b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"f980fa95b3b749b0b63937fe3517d7c3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"1.5.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"1.5.0","_view_name":"HTMLView","description":"","description_tooltip":null,"layout":"IPY_MODEL_f3cd3445a973409083431a24e56a5b4b","placeholder":"​","style":"IPY_MODEL_be9d2c16016b441abecc5d616b54d99e","value":" 5148/5148 [00:00&lt;00:00, 288158.29it/s]"}},"fcb5b69306804aa38a4dd6892c2b460b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"1.2.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"1.2.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"overflow_x":null,"overflow_y":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}}},"version_major":2,"version_minor":0}
</script>
