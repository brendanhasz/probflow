API
===

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   distributions
   parameters
   modules
   models
   callbacks
   data
   applications
   utils


The ProbFlow API consists of four primary modules:

* :mod:`.distributions`
* :mod:`.parameters`
* :mod:`.modules`
* :mod:`.models`

as well as a module which contains pre-built models for standard tasks:

* :mod:`.applications`

and two other modules which allow for customization of the training process:

* :mod:`.callbacks`
* :mod:`.data`

The :mod:`.distributions` module contains classes to instantiate probability
distributions, which describe the likelihood of either a parameter or a
datapoint taking any given value.  Distribution objects are used to represent
the predicted probability distribution of the data, and also the parameters'
posteriors and priors.

The :mod:`.parameters` module contains classes to instantiate parameters,
which are values that characterize the behavior of a model. When fitting a
model, we want to find the values of the parameters which best allow the model
to explain the data. However, with Bayesian modeling we want not only to find
the single best value for each parameter, but a probability distribution which
describes how likely any given value of a parameter is to be the best or true
value.

The :mod:`.modules` module contains the :class:`.Module` abstract base class.
Modules are building blocks which can be used to create probabilistic models.
They can contain parameters, other information, and even other Modules. They
take Tensor(s) as input, perform some computation them, and output a Tensor.
A good example of a Module is a neural network layer, because it needs to
contain parameters (the weights and biases), and it takes one Tensor as input
and outputs a different Tensor.  The :mod:`.modules` module also contains
several specific types of Modules, such as a :class:`.Dense` neural network
layer, a :class:`.BatchNormalization` layer and an :class:`.Embedding` layer.

The :mod:`.models` module contains abstract base classes for Models.  Unlike
Modules, Models encapsulate an entire probabilistic model: they take Tensor(s)
as input and output a probability distribution.  Like Modules, they can
contain Parameters and Modules.  They can be fit to data, and have many
methods for inspecting the quality of the fit.

The :mod:`.applications` module contains pre-built models for standard
applications (e.g. linear regression, logistic regression, and multilayer
dense neural networks) which are ready to be fit.

The :mod:`.callbacks` module contains the :class:`.Callback` abstract base
class, and several specific types of callbacks which can be used to control
the training process, record metrics or parameter values over the course of
training, stop training early, etc.

The :mod:`.data` module contains the :class:`.DataGenerator` class,
which can be used to feed data during training when the dataset is too large
to fit into memory.

The :mod:`.utils` module contains various utilities, generally not intended to
be used by users, such as abstract base classes, casting functions, settings,
etc.
