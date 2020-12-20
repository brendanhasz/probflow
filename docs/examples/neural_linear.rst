Neural Linear Model |Colab Badge|
=================================

.. |Colab Badge| image:: ../img/colab-badge.svg
    :target: TODO

.. include:: ../macros.hrst

The neural linear model is an efficient way to get posterior samples and
uncertainty out of a regular neural network.  Basically, you just slap a
Bayesian linear regression on top of the last hidden layer of a regular
(non-Bayesian, non-probabilistic) neural network.  It was first proposed for
use with Bayesian optimization ([Snoek et al.,
2015](https://arxiv.org/abs/1502.05700)), but has applications in reinforcement
learning (see [Riquelme et al., 2018](https://arxiv.org/abs/1802.09127) and
[Azizzadenesheli and Anandkumar, 2019](https://arxiv.org/abs/1802.04412)),
active learning, AutoML ([Zhou and Precioso,
2019](https://arxiv.org/abs/1904.00577)), and just Bayesian regression problems
in general ([Ober and Rasmussen, 2019](https://arxiv.org/abs/1912.08416)).

Bayesian linear regressions have a closed form solution, so the usual approach
for training a neural linear model is to first train the neural network to
minimize the (for example) mean squared error.  Then in a second step, compute
the closed-form solution to the Bayesian linear regression regressing the last
hidden layer's activations onto the target variable.  Here we'll just use
variational inference to train the neural network and Bayesian regression
together end-to-end.

ProbFlow's :class:`.Dense`, :class:`.DenseNetwork`, and
:class:`.DenseRegression` classes take a `probabilistic` keyword argument,
which when `True` (the default), uses parameters which are probabilistic (i.e.,
they use Normal distributions as the variational distributions).  But when the
`probabilistic` kwarg is set to `False`, then the parameters are totally
non-probabilistic (i.e. a Deterministic distribution, aka a
[Dirac function](https://en.wikipedia.org/wiki/Dirac_delta_function)).  So,
with `probabilistic = False`, ProbFlow won't model any uncertainty as to those
parameters' values (like your run-of-the-mill non-Bayesian neural network
would).

So, to create a neural linear model, we can just create a regular
non-probabilistic neural network using :class:`.DenseNetwork` with
`probabilistic = False`, and then perform a Bayesian linear regression on top
of the final hidden layer (we'll also predict the noise error to allow for
heteroscedasticity).

.. code-block:: python3

    class NeuralLinear(pf.ContinuousModel):

        def __init__(self, dims):
            self.net = pf.DenseNetwork(dims, probabilistic=False)
            self.loc = pf.Dense(dims[-1], 1)  # probabilistic=True by default
            self.std = pf.Dense(dims[-1], 1)  # probabilistic=True by default

        def __call__(self, x):
            h = tf.nn.relu(self.net(x))
            return pf.Normal(self.loc(h), tf.math.softplus(self.std(h)))

    model = NeuralLinear([x.shape[1], 256, 128, 64, 32])
    model.fit(x, y)


TODO: compare the neural linear model to a fully bayesian network, fit to NYC
taxi data (predict fare $ from O/D lat/lng and time), compare fit times (wall
time to same val error - neural linear hopefully faster?), accuracy, and also
uncertainty calibration metrics @ that point (hopefully they're similar?  See
https://github.com/google/uncertainty-metrics, though most of those look like
categorical metrics and I'm more interested in continuous ones...  Been
thinking that the area between the ideal and true calibration curves (so, get
the cumulative probability of the true value along the predicted probability
distribution, sort em by that value, and plot x=1:N, y=those cdf values, then
get the area between that curve and [0,0] to [1,1]).  If calibration is
perfect, score will be 0, larger values are worse (absolute worst value is 1 -
but dunno how that would even happen).


References
----------

* Jasper Snoek, Oren Rippel, Kevin Swersky, Ryan Kiros, Nadathur Satish, Narayanan Sundaram, Md. Mostofa Ali Patwary, Prabhat, Ryan P. Adams.  [Scalable Bayesian Optimization Using Deep Neural Networks](https://arxiv.org/abs/1502.05700), 2015
* Carlos Riquelme, George Tucker, and Jasper Snoek. [Deep Bayesian Bandits Showdown](https://arxiv.org/abs/1802.09127), 2018
* Sebastian W. Ober and Carl Edward Rasmussen. [Benchmarking the Neural Linear Model for Regression](https://arxiv.org/abs/1912.08416), 2019
* Kamyar Azizzadenesheli and Animashree Anandkumar. [Efficient Exploration through Bayesian Deep Q-Networks](https://arxiv.org/abs/1802.04412), 2019
* Weilin Zhou and Frederic Precioso. [Adaptive Bayesian Linear Regression for Automated Machine Learning](https://arxiv.org/abs/1904.00577), 2019

