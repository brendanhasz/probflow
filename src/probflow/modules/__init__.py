"""
Modules are objects which take Tensor(s) as input, perform some computation on
that Tensor, and output a Tensor.  Modules can create and contain |Parameters|.
For example, neural network layers are good examples of a |Module|, since
they store parameters, and use those parameters to perform a computation
(the forward pass of the data through the layer).

* :class:`.Module` - abstract base class for all modules
* :class:`.Dense` - fully-connected neural network layer
* :class:`.DenseNetwork` - a multi-layer dense neural network module
* :class:`.Sequential` - apply a list of modules sequentially
* :class:`.BatchNormalization` - normalize data per batch
* :class:`.Embedding` - embed categorical data in a lower-dimensional space

----------

"""


__all__ = [
    "Module",
    "Dense",
    "DenseNetwork",
    "Sequential",
    "BatchNormalization",
    "Embedding",
]


from .batch_normalization import BatchNormalization
from .dense import Dense
from .dense_network import DenseNetwork
from .embedding import Embedding
from .module import Module
from .sequential import Sequential
