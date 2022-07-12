+++
title = "Overarching concept of deep learning"
description = "Zoom"
colordes = "#e86e0a"
slug = "02_concept"
weight = "2"
+++

Neural networks learn by adjusting their parameters automatically in an iterative manner. This is derived from {{<a "https://en.wikipedia.org/wiki/Arthur_Samuel" "Arthur Samuel">}}'s concept.

It is important to get a good understanding of this process. Let's go over it step by step.
{{<br size="4">}}

### Decide on an architecture

{{<imgbshadow src="/img/diag_white/diag_01.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

The architecture won't change during training. This is set.

### Set some initial parameters

{{<imgbshadow src="/img/diag_white/diag_02.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

You can initialize them randomly or get much better ones through transfer learning.

While the parameters are also part of the model, those will change during training.

### Get some labelled data

{{<imgbshadow src="/img/diag_white/diag_03.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

### Make sure to keep some data for testing

{{<imgbshadow src="/img/diag_white/diag_04.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

Those data won't be used for training the model.

### Train data and parameters are passed through the architecture

{{<imgbshadow src="/img/diag_white/diag_05.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

The train data is the input and the process of calculating the output is the forward pass.

### The output of the model are predictions

{{<imgbshadow src="/img/diag_white/diag_06.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

### Compare those predictions to the train labels

{{<imgbshadow src="/img/diag_white/diag_07.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

Since our data was labelled, we know what the true outputs are.

{{<imgbshadow src="/img/diag_white/diag_08.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

### Calculate train loss

The deviation of our predictions from the true outputs gives us a measure of training loss.

{{<imgbshadow src="/img/diag_white/diag_09.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

### Parameters adjustement

{{<imgbshadow src="/img/diag_white/diag_10.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

The parameters get automatically adjusted to reduce the training loss through the mechanism of backpropagation.

This is the actual training part.

This process is repeated many times.

### From model to program

{{<imgbshadow src="/img/diag_white/diag_11.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

Remember that the model architecture is fixed, but that the parameters change at each iteration of the training process.

When the training is over, we stop changing the parameters. Those are now also fixed and our model is thus a classic program.

{{<imgbshadow src="/img/diag_white/diag_12.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

### Evaluating the model

{{<imgbshadow src="/img/diag_white/diag_13.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

We can now use the testing set (which was never used to train the model) to evaluate our model: if we pass the test data through our program, we get some predictions that we can compare to the test labels (which are the true outputs).

This gives us the test loss.

### Use the model

{{<imgbshadow src="/img/diag_white/diag_14.png" margin="rem" title="" width="70%" line-height="0rem">}}
{{</imgbshadow>}}

Now that we have a program, we can use it on unlabelled inputs to get what we really want: unknown outputs. This is when we put our model to actual use.

## Comments & questions
