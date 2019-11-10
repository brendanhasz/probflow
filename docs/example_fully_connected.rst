.. _example_fully_connected:

Fully-connected Neural Network
==============================

|Colab Badge|

.. |Colab Badge| image:: img/colab-badge.svg
    :target: https://colab.research.google.com/drive/1AgvMHWBAEUpIcrTFY-VhvMGD_RrEKAqG

.. include:: macros.hrst

.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    rand = lambda *x: np.random.rand(*x).astype('float32')
    randn = lambda *x: np.random.randn(*x).astype('float32')
    zscore = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)

    import probflow as pf


Purely linear regressions aren't able handle complex nonlinear relationships
between predictors and target variables - for that kind of data we can use
neural networks!  Regular neural networks simply provide point estimates, but
Bayesian neural networks (BNNs) give us both estimates and uncertainty
information.  BNNs also have strong regularization "built-in" (which comes
from not only the priors, but also from from the sampling performed during
stochastic variational inference).  This makes it much harder for BNNs to
overfit than regular neural networks.

Let's create some nonlinear data to test our neural networks on.  Later we'll fit a network to some real-world data, but for now let's just use this toy dataset:


.. code-block:: python3

    # Create the data
    N = 1024
    x = 10*rand(N, 1)-5
    y = np.sin(x)/(1+x*x) + 0.05*randn(N, 1)

    # Normalize
    x = zscore(x)
    y = zscore(y)

    # Plot it
    plt.plot(x, y, '.')


.. image:: img/examples/fully_connected/output_5_0.svg
   :width: 80 %
   :align: center



Building a Neural Network Manually
----------------------------------

First we'll see how to manually create a Bayesian neural network with ProbFlow
from "scratch", to illustrate how to use the :class:`.Module` class, and to
see why it's so handy to be able to define components from which you can build
a larger model.  Then later we'll use ProbFlow's pre-built modules which make
creating neural networks even easier.

Let's create a module which represents just a single fully-connected layer
(aka a "dense" layer).  This layer takes a vector :math:`\mathbf{x}` (of 
length :math:`N_i`), and outputs a vector of length :math:`N_o`.  It 
multiplies the input by its weights (:math:`\mathbf{W}`, a 
:math:`N_i \times N_o` matrix of learnable parameters), and adds a bias
(:math:`\mathbf{b}`, a :math:`N_o`-length vector of learnable parameters).

.. math::

    \text{DenseLayer}(\mathbf{x}) = \mathbf{x}^\top \mathbf{W} + \mathbf{b}

To use ProbFlow to create a module which represents this layer and creates and
keeps track of the weight and bias parameters, create a class which inherits
:class:`.Module`:

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            import tensorflow as tf

            class DenseLayer(pf.Module):

                def __init__(self, d_in, d_out):
                    self.w = pf.Parameter([d_in, d_out])
                    self.b = pf.Parameter([1, d_out])

                def __call__(self, x):
                    return x @ self.w() + self.b()

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            import torch

            class DenseLayer(pf.Module):

                def __init__(self, d_in, d_out):
                    self.w = pf.Parameter([d_in, d_out])
                    self.b = pf.Parameter([1, d_out])

                def __call__(self, x):
                    x = torch.tensor(x)
                    return x @ self.w() + self.b()


Side note: we've used ``@``, the 
`infix operator for matrix multiplication <https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-465>`_.

Having defined a single layer, it’s much easier to define another 
:class:`.Module` which stacks several of those layers together.  This module 
will represent an entire sub-network of sequential fully connected layers,
with 
`ReLU activation <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_
functions in between each (but no activation after the final layer).  In 
``__init__``, this new module creates and contains several of the 
``DenseLayer`` modules we defined above.


.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            class DenseNetwork(pf.Module):

                def __init__(self, dims):
                    Nl = len(dims)-1 #number of layers
                    self.layers = [DenseLayer(dims[i], dims[i+1]) for i in range(Nl)]
                    self.activations = (Nl-1)*[tf.nn.relu] + [lambda x: x]

                def __call__(self, x):
                    for i in range(len(self.activations)):
                        x = self.layers[i](x)
                        x = self.activations[i](x)
                    return x

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            class DenseNetwork(pf.Module):

                def __init__(self, dims):
                    Nl = len(dims)-1 #number of layers
                    self.layers = [DenseLayer(dims[i], dims[i+1]) for i in range(Nl)]
                    self.activations = (Nl-1)*[torch.nn.ReLU()] + [lambda x: x]

                def __call__(self, x):
                    x = torch.tensor(x)
                    for i in range(len(self.activations)):
                        x = self.layers[i](x)
                        x = self.activations[i](x)
                    return x


The first thing to notice here is that |Modules| can contain other |Modules|!
This allows you to construct models using hierarchical building blocks, making
testing and debugging of your models much easier, and encourages code reuse.

Also note that we've used TensorFlow (or PyTorch) code within the model!
ProbFlow lets you mix and match ProbFlow operations and objects with operations
from the :ref:`backend you've selected <ug_backend>`.

Finally, we can create an actual |Model| which uses the network Module we've
just created.  This model consists of a normal distribution whose mean is predicted by the neural network.  Note that while the ``__call__`` methods of 
the  Modules above returned tensors, the ``__call__`` method of the Model
below returns a *probability distribution*!


.. code-block:: python3

    class DenseRegression(pf.ContinuousModel):
        
        def __init__(self, dims):
            self.net = DenseNetwork(dims)
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(self.net(x), self.s())


Then we can instantiate the model.  We'll create a fully-connected Bayesian
neural network with two hidden layers, each having 32 units.  The first
element of the list passed to the constructor is the number of features (in
this case just one: :math:`x`), and the last element is the number of target
dimensions (in this case also just one: :math:`y`).


.. code-block:: python3

    model = DenseRegression([1, 32, 32, 1])


Then we can fit the network to the data!


.. code-block:: python3

    model.fit(x, y, epochs=1000, lr=0.02)


The fit network can make predictions:


.. code-block:: python3

    # Test points to predict
    x_test = np.linspace(min(x), max(x), 101).astype('float32').reshape(-1, 1)

    # Predict them!
    preds = model.predict(x_test)

    # Plot it
    plt.plot(x, y, '.', label='Data')
    plt.plot(x_test, preds, 'r', label='Predictions')


.. image:: img/examples/fully_connected/output_17_0.svg
   :width: 80 %
   :align: center


And because this is a Bayesian neural network, it also gives us uncertainty
estimates.  For example, we can compute the 95% posterior predictive
distribution intervals:

.. code-block:: python3

    # Compute 95% confidence intervals
    lb, ub = model.predictive_interval(x_test, ci=0.95)

    # Plot em!
    plt.fill_between(x_test[:, 0], lb[:, 0], ub[:, 0], 
                     alpha=0.2, label='95% ci')
    plt.plot(x, y, '.', label='Data')


.. image:: img/examples/fully_connected/output_19_0.svg
   :width: 80 %
   :align: center



Using the Dense and Sequential Modules
--------------------------------------

ProbFlow comes with some ready-made modules for creating fully-connected 
neural networks.  The :class:`.Dense` module handles creating the weight and
bias parameters, and the :class:`.Sequential` module takes a list of modules
or callables and pipes the output of each into the input of the next.

Using these two modules, we can define the same neural network as above much
more easily:

.. tabs::

    .. group-tab:: TensorFlow
    
        .. code-block:: python3

            class DenseRegression(pf.Model):
                
                def __init__(self, d_in):
                    self.net = pf.Sequential([
                        pf.Dense(d_in, 32),
                        tf.nn.relu,
                        pf.Dense(32, 32),
                        tf.nn.relu,
                        pf.Dense(32, 1),
                    ])
                    self.s = pf.ScaleParameter()

                def __call__(self, x):
                    return pf.Normal(self.net(x), self.s())

    .. group-tab:: PyTorch
    
        .. code-block:: python3

            class DenseRegression(pf.Model):
                
                def __init__(self, d_in):
                    self.net = pf.Sequential([
                        pf.Dense(d_in, 32),
                        torch.nn.ReLU(),
                        pf.Dense(32, 32),
                        torch.nn.ReLU(),
                        pf.Dense(32, 1),
                    ])
                    self.s = pf.ScaleParameter()

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Normal(self.net(x), self.s())

Then we can instantiate and fit the network similarly to before:

.. code-block:: python3

    model = DenseRegression(1)
    model.fit(x, y)



Using the DenseNetwork Module
-----------------------------

The :class:`.DenseNetwork` module can be used to automatically create
sequential models of Dense layers with activations in between (by default,
rectified linear activations). Just pass the number of dimensions per dense
layer as a list, and :class:`.DenseNetwork` will create a fully-connected
neural network with the corresponding number of units, rectified linear
activation functions in between, and no activation function after the final
layer.

For example, to create the same model as above with :class:`.DenseNetwork`
(but without having to write the component modules yourself):

    
.. code-block:: python3

    class DenseRegression(pf.Model):
        
        def __init__(self, dims):
            self.net = pf.DenseNetwork(dims)
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(self.net(x), self.s())




Using the DenseRegression or DenseClassifier applications
---------------------------------------------------------

The :class:`.DenseNetwork` module automatically creates sequential dense
layers, but it doesn't include an observation distribuiton.  To create the
same model as before (a multilayer network which predicts the mean of a 
normally-distributed observation distribution), use the 
:class:`.DenseRegression` application:


.. code-block:: python3

    model = pf.DenseRegression([1, 32, 32, 1])
    model.fit(x, y)


To instead use a dense network to perform classification (where the
observation distribution is a categorical distribtuion instead of a normal
distribution), use the :class:`.DenseClassifier` application.

For example, to create a Bayesian neural network (with two hidden layers
containing 32 units each) to perform classification between ``Nc`` categories:


.. code-block:: python3

    # Nf = number of features
    # Nc = number of categories in target

    model = pf.DenseClassifier([Nf, 32, 32, Nc])
    model.fit(x, y)



Fitting a large network to a large dataset
------------------------------------------

TODO: cool but does it scale.  Use a `dual-headed net to predict taxi trip times from the NYC taxi dataset <https://brendanhasz.github.io/2019/07/23/bayesian-density-net.html>`_
