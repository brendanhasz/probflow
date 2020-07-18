"""Tests the statistical accuracy of fitting some distributions"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import probflow as pf

tfd = tfp.distributions


def is_close(a, b, th=1e-5):
    """Check two values are close"""
    return np.abs(a - b) < th


def test_fit_normal():
    """Test fitting a normal distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    mu = np.random.randn()
    sig = np.exp(np.random.randn())
    x = np.random.randn(N, 1).astype("float32")
    x = x * sig + mu

    class NormalModel(pf.Model):
        def __init__(self):
            self.mu = pf.Parameter(name="mu")
            self.sig = pf.ScaleParameter(name="sig")

        def __call__(self):
            return pf.Normal(self.mu(), self.sig())

    # Create and fit model
    model = NormalModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("mu")
    assert lb < mu
    assert ub > mu
    assert is_close(mu, model.posterior_mean("mu"), th=0.2)

    # Check inferences for std are correct
    lb, ub = model.posterior_ci("sig")
    assert lb < sig
    assert ub > sig
    assert is_close(sig, model.posterior_mean("sig"), th=0.2)


def test_fit_studentt():
    """Test fitting a student t distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    mu = np.random.randn(1).astype("float32")
    sig = np.exp(np.random.randn(1)).astype("float32")
    df = np.array([1.0]).astype("float32")
    x = tfd.StudentT(df, mu, sig).sample(N).numpy()

    class StudenttModel(pf.Model):
        def __init__(self):
            self.mu = pf.Parameter(name="mu")
            self.sig = pf.ScaleParameter(name="sig")

        def __call__(self):
            return pf.StudentT(df, self.mu(), self.sig())

    # Create and fit model
    model = StudenttModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("mu")
    assert lb < mu
    assert ub > mu
    assert is_close(mu, model.posterior_mean("mu"), th=0.2)

    # Check inferences for std are correct
    lb, ub = model.posterior_ci("sig")
    assert lb < sig
    assert ub > sig
    assert is_close(sig, model.posterior_mean("sig"), th=0.2)


def test_fit_cauchy():
    """Test fitting a cauchy distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    mu = np.random.randn(1).astype("float32")
    sig = np.exp(np.random.randn(1)).astype("float32")
    x = tfd.Cauchy(mu, sig).sample(N).numpy()

    class CauchyModel(pf.Model):
        def __init__(self):
            self.mu = pf.Parameter(name="mu")
            self.sig = pf.ScaleParameter(name="sig")

        def __call__(self):
            return pf.Cauchy(self.mu(), self.sig())

    # Create and fit model
    model = CauchyModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("mu")
    assert lb < mu
    assert ub > mu
    assert is_close(mu, model.posterior_mean("mu"), th=0.2)

    # Check inferences for std are correct
    lb, ub = model.posterior_ci("sig")
    assert lb < sig
    assert ub > sig
    assert is_close(sig, model.posterior_mean("sig"), th=0.2)


def test_fit_gamma():
    """Test fitting a gamma distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    alpha = np.array([1.0]).astype("float32")
    beta = np.array([1.0]).astype("float32")
    x = tfd.Gamma(alpha, beta).sample(N).numpy().astype("float32")

    class GammaModel(pf.Model):
        def __init__(self):
            self.alpha = pf.PositiveParameter(name="alpha")
            self.beta = pf.PositiveParameter(name="beta")

        def __call__(self):
            return pf.Gamma(self.alpha(), self.beta())

    # Create and fit model
    model = GammaModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("alpha")
    assert lb < alpha
    assert ub > alpha
    assert is_close(alpha, model.posterior_mean("alpha"), th=0.2)

    # Check inferences for std are correct
    lb, ub = model.posterior_ci("beta")
    assert lb < beta
    assert ub > beta
    assert is_close(beta, model.posterior_mean("beta"), th=0.2)


def test_fit_bernoulli():
    """Test fitting a bernoulli distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    prob = 0.7
    x = (tf.random.uniform([N]) < prob).numpy().astype("float32")

    class BernoulliModel(pf.Model):
        def __init__(self):
            self.prob = pf.BoundedParameter(name="prob")

        def __call__(self):
            return pf.Bernoulli(probs=self.prob())

    # Create and fit model
    model = BernoulliModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("prob")
    assert lb < prob
    assert ub > prob
    assert is_close(prob, model.posterior_mean("prob"), th=0.1)


def test_fit_categorical():
    """Test fitting a categorical distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    probs = [0.3, 0.2, 0.5]
    x = tfd.Categorical(probs=probs).sample(N).numpy().astype("float32")

    class CategoricalModel(pf.Model):
        def __init__(self):
            self.probs = pf.DirichletParameter(k=3, name="probs")

        def __call__(self):
            return pf.Categorical(probs=self.probs())

    # Create and fit model
    model = CategoricalModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("probs")
    for i in range(len(probs)):
        assert lb[i] < probs[i]
        assert ub[i] > probs[i]
    assert all(is_close(probs, model.posterior_mean("probs"), th=0.05))


def test_fit_poisson():
    """Test fitting a poisson distribution"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 10000
    rate = 10
    x = tf.random.poisson([N], rate).numpy()

    class PoissonModel(pf.Model):
        def __init__(self):
            self.rate = pf.PositiveParameter(name="rate")

        def __call__(self):
            return pf.Poisson(self.rate())

    # Create and fit model
    model = PoissonModel()
    model.fit(x, batch_size=100, epochs=1000, lr=1e-2)

    # Check inferences for mean are correct
    lb, ub = model.posterior_ci("rate")
    assert lb < rate
    assert ub > rate
    assert is_close(rate, model.posterior_mean("rate"), th=1.0)
