# Training a fully connected network from scratch using numpy


References in the respective scripts
## Synopsis

Was written specifically to train for MNIST digit classification. The code is for CPU. 

## Usage and Features

The training can be initiated through run.sh. Various hyperparameters can be set in that file like
* Learning Rate
* Number of hidden layers
* Momentum
* Sizes of all layers
* Batch Size
* Activation Function : tanh and sigmoid
* Optimizer : Adam, NAG, SGD, Momentum
* Learning Rate annealing 
* Loss Function : Cross Entropy, MSE

Data Augmentation ( Elastic Distortion is performed)
## Dependancies 

* Numpy
* cPickle
* Scipy
* Matplotlib

Best Accuracy : 98.25%

