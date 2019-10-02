.. _example_fully_connected:

Fully-connected Neural Network
==============================

.. include:: macros.hrst

TODO: intro, link to colab w/ these examples

.. contents:: Outline

TODO: this shows how to do it manually and why it's nice to use Modules as building blocks for large models.  However, it's even easier to do this using ProbFlow's Dense + Sequential modules, or the DenseRegression model, as we'll see in the following sections.


Manually
--------

TODO: manually input -> 128 units -> 64 units -> 1 unit w/ no activation -> normal observation dist

TODO: math

TODO: diagram

First we'll make a module which represents a single fully-connected layer:

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            import probflow as pf
            import tensorflow as tf

            class DenseLayer(pf.Module):

                def __init__(self, d_in, d_out):
                    self.w = pf.Parameter([d_in, d_out])
                    self.b = pf.Parameter([1, d_out])

                def __call__(self, x):
                    return x @ self.w() + self.b()

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            import probflow as pf
            import torch

            class DenseLayer(pf.Module):

                def __init__(self, d_in, d_out):
                    self.w = pf.Parameter([d_in, d_out])
                    self.b = pf.Parameter([d_out, 1])

                def __call__(self, x):
                    x = torch.tensor(x)
                    return x @ self.w() + self.b()


Note that we've used ``@``, the 
`infix operator for matrix multiplication <https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-465>`_.

Having defined a single layer, it's much easier to define another |Module| which 
stacks several of those layers, with activation functions in between each:

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            class DenseNetwork(pf.Module):
                
                def __init__(self, dims):
                    Nl = len(dims)-1
                    self.layers = [DenseLayer(dims[i], dims[i+1]) for i in range(Nl)]
                    self.activations = [tf.nn.relu for i in range(Nl)]
                    self.activations[-1] = lambda x: x


                def __call__(self, x):
                    for i in range(len(self.layers)):
                        x = self.layers[i](x)
                        x = self.activations[i](x)
                    return x

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            class DenseNetwork(pf.Module):
                
                def __init__(self, dims):
                    Nl = len(dims)-1
                    self.layers = [DenseLayer(dims[i], dims[i+1]) for i in range(Nl)]
                    self.activations = [torch.nn.ReLU() for i in range(Nl)]
                    self.activations[-1] = lambda x: x


                def __call__(self, x):
                    x = torch.tensor(x)
                    for i in range(len(self.layers)):
                        x = self.layers[i](x)
                        x = self.activations[i](x)
                    return x


The first thing to notice here is that |Modules| can contain other |Modules|!
This allows you to construct models using hierarchical building blocks, making
testing and debugging of your models much easier.

Also note that we've used TensorFlow code within the model!  ProbFlow lets you
mix and match ProbFlow operations and objects with operations from the backend 
you've selected.  |TensorFlow| is the default backend, but if we had wanted to
use |PyTorch| (see :ref:`ug_backend`), we could have used PyTorch's relu
function:

.. code-block:: python3

    import torch

    ...
    self.activations = [torch.nn.ReLU() for i in range(Nl)]
    ...

Finally, we can create a |Model| which uses the network |Module| we've just created.  This model consists of a normal distribution whose mean is predicted
by the neural network:

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            class DenseRegression(pf.Model):
                
                def __init__(self, dims):
                    self.net = DenseNetwork(dims)
                    self.s = pf.ScaleParameter()

                def __call__(self, x):
                    return pf.Normal(self.net(x), self.s())

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            class DenseRegression(pf.Model):
                
                def __init__(self, dims):
                    self.net = DenseNetwork(dims)
                    self.s = pf.ScaleParameter()

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Normal(self.net(x), self.s())

TODO: then can fit the net

.. code-block:: python3

    model = DenseRegression([5, 128, 64, 1])
    model.fit(x, y)


Using the Dense and Sequential Modules
--------------------------------------

TODO: the Dense module handles creating the variables for you, and the Sequential module takes a list of modules or callables and pipes the output of each into the input of the next

.. tabs::

    .. group-tab:: TensorFlow
    
        .. code-block:: python3

            class DenseRegression(pf.Model):
                
                def __init__(self):
                    self.net = pf.Sequential([
                        pf.Dense(5, 128),
                        tf.nn.relu,
                        pf.Dense(128, 64),
                        tf.nn.relu,
                        pf.Dense(64, 1),
                    ])
                    self.s = pf.ScaleParameter()

                def __call__(self, x):
                    return pf.Normal(self.net(x), self.s())

    .. group-tab:: PyTorch
    
        .. code-block:: python3

            class DenseRegression(pf.Model):
                
                def __init__(self):
                    self.net = pf.Sequential([
                        pf.Dense(5, 128),
                        torch.nn.ReLU(),
                        pf.Dense(128, 64),
                        torch.nn.ReLU(),
                        pf.Dense(64, 1),
                    ])
                    self.s = pf.ScaleParameter()

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Normal(self.net(x), self.s())

TODO: then can fit the net

.. code-block:: python3

    model = DenseRegression([5, 128, 64, 1])
    model.fit(x, y)


Using the DenseRegression or DenseClassifier applications
---------------------------------------------------------

TODO: the DenseNet model automatically creates sequential dense layers, but NOT an observation distribution, default is relu activation but no activation for last layer


TODO: DenseRegression

.. code-block:: python3

    model = pf.DenseRegression([5, 128, 64, 1])
    model.fit(x, y)


TODO: DenseClassifier

.. code-block:: python3

    # TODO make dataset w/ categorical output

    model = pf.DenseClassifier([5, 128, 64, 1])
    model.fit(x, y)
