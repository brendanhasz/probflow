API
===

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   api_core
   api_parameters
   api_layers
   api_distributions
   api_models


The ProbFlow API consists of five modules:

* :mod:`.core`
* :mod:`.parameters`
* :mod:`.layers`
* :mod:`.distributions`
* :mod:`.models`

The :mod:`.core` module contains abstract base classes (ABCs) for all of
ProbFlow's classes.  In other words, none of the classes in the :mod:`.core` 
module can be instantiated - they only exist to implement functionality 
common to the classes which inherit them.

The :mod:`.parameters` module contains classes to instantiate parameters, which are values that characterize the behavior of a model. When fitting a model, we want to find the values of the parameters which best allow the model to explain the data. However, with Bayesian modeling we want not only to find the single best value for each parameter, but a probability distribution which describes how likely any given value of a parameter is to be the best or true value.

The :mod:`.layers` module contains classes to instantiate layers, which are objects that take one or more tensors as input arguments, perform some computation on them, and output a single tensor. The input tensor(s) can be the input data, constants, or the outputs of other layers. Layers can additionally take keyword arguments which control how the computation is performed.

The :mod:`.distributions` module contains classes to instantiate probability distributions, which describe the likelihood of either a parameter or a datapoint taking any given value.  Distribution objects are used to represent the predicted probability distribution of the data, and also the parameters' posteriors and priors.

Finally, the :mod:`.models` module contains several common ready-made models, such as a linear regression, a logistic regression, and a multilayer fully-connected neural network.
