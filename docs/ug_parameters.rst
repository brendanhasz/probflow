.. _ug_parameters:

Parameters
==========

.. include:: macros.hrst

TODO: parameters of the model, etc

TODO: update for v2


Creating a Parameter
--------------------

TODO: creating, naming, and setting the shape of parameters


Specifying the Variational Posterior
------------------------------------

TODO: setting ``posterior`` kwarg to set the variational posterior


Setting the Prior
-----------------

The default prior on a |Parameter| is a :class:`.Normal` distribution with a
mean of 0 and a standard deviation of 1.  However, we can manually set the
prior on any |Parameter| to be any |Distribution| (with any parameters).
The only limitation is that the backend must be able to analytically compute
the Kullback–Leibler divergence between the prior and the posterior (which is
usually possible as long as you use the same type of distribution for both the
prior and posterior).

To set the prior on a parameter, use the ``prior`` keyword argument to a 
|Parameter| when initializing that parameter.  For example, if instead of the
default, we want our parameter to have a :class:`.Normal` prior which has a
mean of 1 and a scale of 2, we can set this prior by using the ``prior``
keyword argument:

.. code-block:: python3

    new_param = pf.Parameter(prior=pf.Normal(1, 2))


Transforming Parameters
-----------------------

TODO: talk about transform kwarg to constructor.  Parameter w/ lognormal posterior as an example?  Also mention that the transform is only applied to the samples the posterior emits, and NOT to the prior (i.e. the prior and posterior should be in the same non-transformed space)


Setting the variable initializers
---------------------------------

The posterior distributions have one or more variables which determine the 
shape of the distribution (these are the variables which are optimized over
the course of training).  You can set how the values of the variables are
initialized at the beginning of training. The default is to use
`Xavier initialization <http://proceedings.mlr.press/v9/glorot10a.html>`_
(aka Glorot initialization) for the mean of the default Normal posterior
distribution, and a shifted Xavier initializer for the standard deviation
variable.

To use a custom initializer, use the ``initializer`` keyword argument to the
|Parameter| constructor.  Pass a dictionary where the keys are the variable
names and the values are functions which have one argument - the parameter's 
shape - and return a tensor of initial values.  For example, to create a 
matrix of |Parameters| with Normal priors and posteriors, and initialize the
posterior's ``loc`` (the mean) variable by drawing values from a normal 
distribution, and the ``scale`` (the standard deviation) parameter with all
ones:

.. code-block:: python3

    import tensorflow as tf

    def randn_fn(shape):
        return tf.random.normal(shape)

    def ones_fn(shape):
        return tf.ones(shape)

    init_dict = {'loc': randn_fn, 'scale': ones_fn}
    new_param = pf.Parameter(initializer=init_dict)


Setting the variable transforms
-------------------------------

TODO



Scale Parameters
----------------

TL;DR: to make a standard deviation parameter, use the 
:class:`.ScaleParameter` class:

.. code-block:: python3

    std_dev = pf.ScaleParameter()

A parameter which comes up often in Bayesian modeling is a "scale" parameter.  For example, the standard deviation (:math:`\sigma`) in a linear regression with normally-distributed noise:

.. math::

    p(y~|~x) = \mathcal{N}(\beta x + \beta_0, ~ \sigma)

This :math:`\sigma` parameter cannot take values below 0, because the standard
deviation cannot be negative.  So, we can't use the default posterior and
prior for a |Parameter| (which is a :class:`.Normal` distribution for the
posterior and ``Normal(0, 1)`` for the prior), because this default allows
negative values.

In Bayesian modeling, the `gamma distribution <https://en.wikipedia.org/wiki/
Gamma_distribution>`_ is often used as a posterior for the 
`precision <https://en.wikipedia.org/wiki/Precision_(statistics)>`_.  The 
precision is the reciprocal of the variance, and so the
`inverse gamma distribution <https://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_ 
can be used as a variational posterior for the variance.

However, many models are parameterized in terms of the standard deviation,
which is the square root of the variance.  So, to create a standard deviation
parameter, we could first construct a variance parameter (:math:`\sigma^2`)
which uses an inverse gamma distribution as its variational posterior:

.. math::

    \sigma^2 \sim \text{InvGamma}(\alpha, \beta)

and then transform this into a standard deviation parameter (:math:`\sigma`):

.. math::

    \sigma = \sqrt{\sigma^2}

This could be accomplished with ProbFlow by
`setting the posterior <#specifying-the-variational-posterior>`_ and
`the prior <#setting-the-prior>`_ to :class:`.InvGamma`, which means we would
also have to specify the initializers and variable transformations
accordingly, and then
`transform the parameter <#transforming-parameters>`_ with a square root:

.. code-block:: python3

    from probflow.utils.initializers import pos_xavier

    std_dev = pf.Parameter(posterior=pf.InvGamma,
                           prior=pf.InvGamma(5, 5),
                           transform=lambda x: tf.sqrt(x),
                           initializer={'concentration': pos_xavier,
                                        'scale': pos_xavier},
                           var_transform={'concentration': tf.nn.softplus,
                                          'scale': tf.nn.softplus})

Since that's such a pain, ProbFlow provides a :class:`.ScaleParameter` class
which automatically creates a parameter with the above variational posterior
and transforms, etc.  This makes it much easier to create a scale parameter:

.. code-block:: python3

    std_dev = pf.ScaleParameter()


Categorical Parameters
----------------------

TODO


Bounded Parameters
------------------

TODO


Positive Parameter
------------------

TODO


Deterministic Parameter
-----------------------

TODO


Dirichlet Parameter
-------------------

TODO

