.. _parameters:

Parameters
==========

.. include:: macros.hrst

TODO: parameters of the model, etc


Creating a Parameter
--------------------

TODO: creating, naming, and setting the shape of parameters


Specifying the Variational Posterior
------------------------------------

TODO: setting ``posterior_fn`` to set the variational posterior


Setting the Prior
-----------------

The default prior on a |Parameter| is a :class:`.Normal` distribution with a mean of 0 and a standard deviation of 1.  However, you can manually set the prior on any |Parameter| to be any |Distribution| you wish (with any parameters).

There are two ways to set the prior on a parameter. The first way is to set the ``prior`` keyword argument to a |Distribution| object when initializing the parameter.  For example, if instead of the default, we want our parameter to have a :class:`.Normal` prior which has a mean of 1 and a scale of 2, we can set this prior by using the ``prior`` keyword argument:

.. code-block:: python

   new_param = Parameter(prior=Normal(1,2))

The second way to set the prior is to use the ``<<`` operator, with this syntax:

.. code-block:: python

   parameter << prior

For example, we could also have set the prior on a parameter to :math:`\mathcal{N}(1, 2)` by:

.. code-block:: python

   new_param = Parameter()
   new_param << Normal(1,2)

Setting the prior via the ``<<`` operator is provided mainly for readability (it's meant to be used like the ``~`` operator in `Stan <https://mc-stan.org/>`_).  It makes code much more readable especially when specifying complex multilevel models.  For example, instead of:

.. code-block:: python

   pop_mu = Parameter()
   pop_std = ScaleParameter()
   mu = Parameter(shape=100, prior=Normal(pop_mu, pop_std))

We could use the ``<<`` operator to specify the same model:

.. code-block:: python

   pop_mu = Parameter()
   pop_std = ScaleParameter()
   mu = Parameter(shape=100)

   mu << Normal(pop_mu, pop_std)


Scale Parameters
----------------

A parameter which comes up often in Bayesian modeling is a "scale" parameter.  For example, the standard deviation (:math:`\sigma`) in a linear regression with normally-distributed noise:

.. math::

    p(y~|~x) = \mathcal{N}(\beta x + \beta_0, ~ \sigma)

This :math:`\sigma` parameter cannot take values below 0, because the standard deviation cannot be negative.  So, we can't use the default posterior and prior for a |Parameter| (which is a :class:`.Normal` distribution for the posterior and :math:`\mathcal{N}(0, 1)` for the prior), because this default allows negative values.

In Bayesian modeling, the `gamma distribution <https://en.wikipedia.org/wiki/Gamma_distribution>`_ is often used as a posterior for the `precision <https://en.wikipedia.org/wiki/Precision_(statistics)>`_.  The precision is the reciprocal of the variance, and so the `inverse gamma distribution <https://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_ can be used as a variational posterior for the variance.

However, many models are parameterized in terms of the standard deviation, which is the square root of the variance.  So, to create a standard deviation parameter, we could first construct a variance parameter (:math:`\sigma^2`) which uses an inverse gamma distribution as its variational posterior:

.. math::

    \sigma^2 \sim \text{InvGamma}(\alpha, \beta)

and then transform this into the standard deviation parameter (:math:`\sigma`):

.. math::

    \sigma = \sqrt{\sigma^2}

This could be accomplished using probflow by `setting the posterior <#specifying-the-variational-posterior>`_ to :class:`.InvGamma`, and then `transforming the parameter <#transforming-parameters>`_ with a square root:

.. code-block:: python

   from probflow import Parameter, InvGamma

   std_dev = Parameter(posterior_fn=InvGamma,
                       transform=lambda x: tf.sqrt(x),
                       inv_transform=lambda x: tf.square(x))

For convenience, ProbFlow provides a :class:`.ScaleParameter` class which automatically creates a parameter with the above variational posterior and transforms.  This makes it much easier to create a scale parameter:

.. code-block:: python

   from probflow import ScaleParameter

   std_dev = ScaleParameter()

By default, :class:`.ScaleParameter` uses a uniform prior.


Transforming Parameters
-----------------------

TODO: talk about transform and inv_transform args to constructor.  Parameter w/ lognormal posterior as an example?  Also mention that the transform is only applied to the samples the posterior emits, and NOT to the prior (i.e. the prior and posterior should be in the same non-transformed space)
