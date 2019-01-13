Fitting a Model
===============

TODO: discuss how models have to end in a distribution and how to fit them, etc


Appendix: Mathematical Details
------------------------------

Variables: :math:`v`

Data: :math:`\mathcal{D}`

Prior: :math:`p(v)`

Likelihood: :math:`p(\mathcal{D}|v)`

Posterior: :math:`p(v|\mathcal{D})`

With variational inference we approximate the posterior for each variable with a "variational posterior distribution" :math:`q`. That variational distribution has some parameters :math:`\theta`.  For example if we use a normal distribution as our variational distribution, it has two parameters (:math:`\mu` and :math:`\sigma`), and so :math:`\theta = \{ \mu, \sigma \}`.

The idea is to find the values of :math:`\theta` such that the difference between :math:`q(v|\theta)` (the variational distribution) and :math:`p(v|\mathcal{D})` (the true posterior distribution) is as small as possible.

If we use Kullback-Leibler divergence as our measure of "difference", then we want to find the best values for our variational distribution parameters (:math:`\hat{\theta}`) which give the lowest KL divergence between the variational distribution and the true posterior:

.. math::

    \hat{\theta} = \arg \min_\theta ~ \text{KL}(~q(v|\theta)~||~p(v|\mathcal{D})~) 

This divergence between the variational and true posteriors is equal to the sum of three terms:

1. the divergence between the prior and the variational distribution
2. the expected log likelihood
3. the log model evidence (probability of the data)

.. math::

    \text{KL}(~q(v|\theta)~||~p(v|\mathcal{D})~) =& \int q(v|\theta) \log \frac{q(v|\theta)}{p(v|\mathcal{D})} dv \\
    ~ & \int q(v|\theta) \log \frac{q(v|\theta) ~ p(\mathcal{D})}{p(\mathcal{D}|v)~p(v)} dv \\
    ~ & \int q(v|\theta) 
        \left( \log \frac{q(v|\theta)}{p(v)} - \log p(\mathcal{D}|v) + \log p(\mathcal{D}) \right) dv \\
    ~ & \int \left( q(v|\theta) 
         \log \frac{q(v|\theta)}{p(v)} - q(v|\theta) \log p(\mathcal{D}|v) + q(v|\theta) \log p(\mathcal{D}) \right) dv \\
    ~ & \int q(v|\theta) \log \frac{q(v|\theta)}{p(v)} dv
        - \int q(v|\theta) \log p(\mathcal{D}|v) dv
        + \int q(v|\theta) \log p(\mathcal{D}) dv \\
    ~ & \int q(v|\theta) \log \frac{q(v|\theta)}{p(v)} dv
        - \int q(v|\theta) \log p(\mathcal{D}|v) dv
        + \log p(\mathcal{D}) \\
    ~ & \text{KL} (~q(v|\theta)~||~p(v)~)
        - \int q(v|\theta) \log p(\mathcal{D}|v) dv 
        + \log p(\mathcal{D}) \\
    ~ & \text{KL} (~q(v|\theta)~||~p(v)~)
        - \mathbb{E}_{q(v|\theta)} [ \log p(\mathcal{D}|v) ] 
        + \log p(\mathcal{D})

The model evidence (:math:`\log p(\mathcal{D})`) is a constant, so in order to minimize the divergence between the variational and true posteriors, we can just minimize the sum of the divergence between the prior and variational distributions, and the expected log likelihood:

.. math::

    \hat{\theta} = \arg \min_\theta ~ \text{KL} (~q(v|\theta)~||~p(v)~) - \mathbb{E}_{q(v|\theta)} [ \log p(\mathcal{D}|v) ]