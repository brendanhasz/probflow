.. _math:

Mathematical Details
====================

.. include:: macros.hrst


ProbFlow fits Bayesian models to data using stochastic variational inference [1]_, specifically by performing "Bayes by backprop" [2]_.

Notation:

- Parameters: :math:`\beta`
- Data: :math:`\mathcal{D}`
- Prior: :math:`p(\beta)`
- Likelihood: :math:`p(\mathcal{D}|\beta)`
- Posterior: :math:`p(\beta|\mathcal{D})`

With variational inference we approximate the posterior for each parameter with a "variational posterior distribution" :math:`q`. That variational distribution has some variables :math:`\theta`.  For example, if we use a normal distribution as our variational distribution, it has two variables: :math:`\mu` and :math:`\sigma`.  So, :math:`\theta = \{ \mu, \sigma \}` and 

.. math::

    q(\beta|\theta) = q(\beta|\mu,\sigma) = \mathcal{N}(\beta | \mu, \sigma)

To "fit" a Bayesian model with this method, we want to find the values of :math:`\theta` such that the difference between :math:`q(\beta|\theta)` (the variational distribution) and :math:`p(\beta|\mathcal{D})` (the true posterior distribution) is as small as possible.

If we use `Kullback-Leibler divergence <http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_ as our measure of "difference", then we want to find the best values for our variational distribution variables (:math:`\hat{\theta}`) which give the lowest KL divergence between the variational distribution and the true posterior:

.. math::

    \hat{\theta} = \arg \min_\theta ~ \text{KL}(~q(\beta|\theta)~||~p(\beta|\mathcal{D})~) 

The problem is, we don't know what the true posterior looks like - that's what we're trying to solve!  Luckily, this divergence between the variational and true posteriors can be broken down into the sum of three terms [3]_:

1. the divergence between the prior and the variational distribution
2. the (negative) expected log likelihood
3. the log model evidence (the probability of the data)

Proof:

.. math::

    \text{KL}(~q(\beta|\theta)~||~p(\beta|\mathcal{D})~) =& \int q(\beta|\theta) \log \frac{q(\beta|\theta)}{p(\beta|\mathcal{D})} d\beta \\
    ~ =& \int q(\beta|\theta) \log \frac{q(\beta|\theta) ~ p(\mathcal{D})}{p(\mathcal{D}|\beta)~p(\beta)} d\beta \\
    ~ =& \int q(\beta|\theta) 
        \left( \log \frac{q(\beta|\theta)}{p(\beta)} - \log p(\mathcal{D}|\beta) + \log p(\mathcal{D}) \right) d\beta \\
    ~ =& \int \left( q(\beta|\theta) 
         \log \frac{q(\beta|\theta)}{p(\beta)} - q(\beta|\theta) \log p(\mathcal{D}|\beta) + q(\beta|\theta) \log p(\mathcal{D}) \right) d\beta \\
    ~ =& \int q(\beta|\theta) \log \frac{q(\beta|\theta)}{p(\beta)} d\beta
        - \int q(\beta|\theta) \log p(\mathcal{D}|\beta) d\beta
        + \int q(\beta|\theta) \log p(\mathcal{D}) d\beta \\
    ~ =& \int q(\beta|\theta) \log \frac{q(\beta|\theta)}{p(v)} d\beta
        - \int q(\beta|\theta) \log p(\mathcal{D}|\beta) d\beta
        + \log p(\mathcal{D}) \\
    ~ =& ~ \text{KL} (~q(\beta|\theta)~||~p(\beta)~)
        - \int q(\beta|\theta) \log p(\mathcal{D}|\beta) dv 
        + \log p(\mathcal{D}) \\
    ~ =& ~ \text{KL} (~q(\beta|\theta)~||~p(\beta)~)
        - \mathbb{E}_{q(\beta|\theta)} [~\log p(\mathcal{D}|\beta)~] 
        + \log p(\mathcal{D})

The model evidence (:math:`\log p(\mathcal{D})`) is a constant, so in order to minimize the divergence between the variational and true posteriors, we can just minimize the right-hand side of the equation, ignoring the model evidence:

.. math::

    \hat{\theta} = \arg \min_\theta ~ \text{KL} (~q(\beta|\theta)~||~p(\beta)~) - \mathbb{E}_{q(\beta|\theta)} [~\log p(\mathcal{D}|\beta)~]

These two terms are known as the "variational free energy", or the (negative) "evidence lower bound" (ELBO).

During optimization, we can analytically compute the divergence between the priors and the variational posteriors (:math:`\text{KL} (~q(\beta|\theta)~||~p(\beta)~)`), assuming this is possible given the types of distributions we used for the prior and posterior (e.g. Normal distributions).  We can estimate the expected log likelihood (:math:`\mathbb{E}_{q(\beta|\theta)} [~\log p(\mathcal{D}|\beta)~]`) by sampling parameter values from the variational distribution for each datapoint in our batch, and then computing the average log likelihood for those samples.  That is, we estimate it via Monte Carlo.

When creating a loss function to maximize the ELBO, we need to be careful about batching.  The above minimization equation assumes all samples are being used, but when using stochastic gradient descent, we have only a subset of the samples at any given time.  So, we need to ensure the contribution of the log likelihood and the KL divergence are scaled similarly.  Since we're using a Monte Carlo estimation of the expected log likelihood anyway, with batching we can still just take the mean log likelihood of our samples as the contribution of the log likelihood term.  However, the divergence term should be applied once per *pass through the data*, so we need to normalize it by the *total number of datapoints*, not by the numeber of datapoints in the batch.  With TensorFlow, this looks like:

.. code-block:: python

    # kl_loss = sum of prior-posterior divergences
    # log_likelihood = mean log likelihood of samples in batch
    # N = number of samples in the entire dataset
    elbo_loss = kl_loss/N - log_likelihood



References
----------
.. [1] Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley.
    Stochastic Variational Inference.
    *Journal of Machine Learning Research* 14:1303−1347, 2013.
    http://jmlr.org/papers/v14/hoffman13a.html
.. [2] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. 
    Weight uncertainty in neural networks. 
    *arXiv preprint*, 2015. http://arxiv.org/abs/1505.05424
.. [3] Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul.
    An Introduction to Variational Methods for Graphical Models.
    *Machine Learning* 37:183, 1999.
    http://doi.org/10.1023/A:1007665907178
