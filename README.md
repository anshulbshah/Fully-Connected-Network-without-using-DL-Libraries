# Training a fully connected network from scratch using numpy

Example for MNIST digit classification on CPU
Supports the following features :

1)Adam, SGD, Momentum, NAG supported
2)Learning rate annealing
3)Gorlot Weight Initialization
4)Cross Entropy, Square error losses
5)Tanh and Sigmoid activation
6)Change number and size of hidden layers
7)Data Augmentation : Elastic Distortion


References in the respective scripts
## Synopsis

Was written specifically to train for MNIST digit classification. The code is for CPU. 

## Usage

The training can be initiated through run.sh. Various hyperparameters can be set in that file like
*Learning Rate
*Number of hidden layers
*Momentum
*Sizes of all layers
*Batch Size
*Activation Function : tanh and sigmoid
*Optimizer : Adam, NAG, SGD, Momentum
*Learning Rate annealing 
*Loss Function : Cross Entropy, MSE

## Dependancies 

*Numpy
*cPickle
*Scipy
*Matplotlib



