Mathematical Details
====================

.. include:: macros.hrst

TODO: intro


Variational Inference
---------------------

TODO: intro

see ref [1]_

Notation:

- Variables: :math:`v`
- Data: :math:`\mathcal{D}`
- Prior: :math:`p(v)`
- Likelihood: :math:`p(\mathcal{D}|v)`
- Posterior: :math:`p(v|\mathcal{D})`

With variational inference we approximate the posterior for each variable with a "variational posterior distribution" :math:`q`. That variational distribution has some parameters :math:`\theta`.  For example if we use a normal distribution as our variational distribution, it has two parameters (:math:`\mu` and :math:`\sigma`).  So, :math:`\theta = \{ \mu, \sigma \}` and 

.. math::

    q(v|\theta) = q(v|\mu,\sigma) = \mathcal{N}(v | \mu, \sigma)

The idea is to find the values of :math:`\theta` such that the difference between :math:`q(v|\theta)` (the variational distribution) and :math:`p(v|\mathcal{D})` (the true posterior distribution) is as small as possible.

If we use `Kullback-Leibler divergence <http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_ as our measure of "difference", then we want to find the best values for our variational distribution parameters (:math:`\hat{\theta}`) which give the lowest KL divergence between the variational distribution and the true posterior:

.. math::

    \hat{\theta} = \arg \min_\theta ~ \text{KL}(~q(v|\theta)~||~p(v|\mathcal{D})~) 

This divergence between the variational and true posteriors can be broken down into the sum of three terms:

1. the divergence between the prior and the variational distribution
2. the (negative) expected log likelihood
3. the log model evidence (the probability of the data)

.. math::

    \text{KL}(~q(v|\theta)~||~p(v|\mathcal{D})~) =& \int q(v|\theta) \log \frac{q(v|\theta)}{p(v|\mathcal{D})} dv \\
    ~ & \int q(v|\theta) \log \frac{q(v|\theta) ~ p(\mathcal{D})}{p(\mathcal{D}|v)~p(v)} dv \\
    ~ =& \int q(v|\theta) 
        \left( \log \frac{q(v|\theta)}{p(v)} - \log p(\mathcal{D}|v) + \log p(\mathcal{D}) \right) dv \\
    ~ =& \int \left( q(v|\theta) 
         \log \frac{q(v|\theta)}{p(v)} - q(v|\theta) \log p(\mathcal{D}|v) + q(v|\theta) \log p(\mathcal{D}) \right) dv \\
    ~ =& \int q(v|\theta) \log \frac{q(v|\theta)}{p(v)} dv
        - \int q(v|\theta) \log p(\mathcal{D}|v) dv
        + \int q(v|\theta) \log p(\mathcal{D}) dv \\
    ~ =& \int q(v|\theta) \log \frac{q(v|\theta)}{p(v)} dv
        - \int q(v|\theta) \log p(\mathcal{D}|v) dv
        + \log p(\mathcal{D}) \\
    ~ =& \text{KL} (~q(v|\theta)~||~p(v)~)
        - \int q(v|\theta) \log p(\mathcal{D}|v) dv 
        + \log p(\mathcal{D}) \\
    ~ =& \text{KL} (~q(v|\theta)~||~p(v)~)
        - \mathbb{E}_{q(v|\theta)} [~\log p(\mathcal{D}|v)~] 
        + \log p(\mathcal{D})

The model evidence (:math:`\log p(\mathcal{D})`) is a constant, so in order to minimize the divergence between the variational and true posteriors, we can just minimize the right-hand side of the equation, ignoring the model evidence:

.. math::

    \hat{\theta} = \arg \min_\theta ~ \text{KL} (~q(v|\theta)~||~p(v)~) - \mathbb{E}_{q(v|\theta)} [~\log p(\mathcal{D}|v)~]

These two terms are known as the "variational free energy", or the (negative) "evidence lower bound" (ELBO).

During optimization, we can analytically compute the divergence between the priors and the variational posteriors (:math:`\text{KL} (~q(v|\theta)~||~p(v)~)`), assuming this is possible given the types of distributions we used for the prior and posterior (e.g. Normal distributions).  We can estimate the expected log likelihood (:math:`\mathbb{E}_{q(v|\theta)} [~\log p(\mathcal{D}|v)~]`) by sampling parameter values from the variational distribution for each datapoint in our batch, and then computing the average log likelihood for those samples.  That is, we estimate it via Monte Carlo.

When creating a loss function to maximize the ELBO, we need to be careful about batching.  The above minimization equation assumes all samples are being used, but when using stochastic gradient descent, we have only a subset of the samples at any given time.  So, we need to ensure the contribution of the log likelihood and the KL divergence are scaled similarly.  Since we're using a Monte Carlo estimation of the expected log likelihood anyway, with batching we can still just take the mean log likelihood of our samples as the contribution of the log likelihood term.  However, the divergence term should be applied once per *pass through the data*, so we need to normalize it by the *total number of datapoints*, not by the numeber of datapoints in the batch.  With TensorFlow, this looks like:

.. code-block:: python

    # kl_loss = sum of prior-posterior divergences
    # log_likelihood = mean log likelihood of samples in batch
    # N = number of samples in the entire dataset
    elbo_loss = kl_loss/N - log_likelihood


Flipout
-------

TODO

see ref [2]_

References
----------
.. [1] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. 
    Weight uncertainty in neural networks. 
    *arXiv preprint*, 2015. http://arxiv.org/abs/1505.05424
.. [2] Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. 
    Flipout: Efficient Pseudo-Independent Weight Perturbations on 
    Mini-Batches. *International Conference on Learning Representations*, 
    2018. http://arxiv.org/abs/1803.04386


