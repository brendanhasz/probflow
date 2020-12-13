"""Tests the probflow.models module when backend = pytorch"""


import numpy as np
import pytest
import torch

import probflow.utils.ops as O
from probflow.data import ArrayDataGenerator
from probflow.distributions import Normal
from probflow.models import *
from probflow.modules import *
from probflow.parameters import *
from probflow.utils.settings import Sampling


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Model_0D():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter(name="Weight")
            self.bias = Parameter(name="Bias")
            self.std = ScaleParameter(name="Std")

        def __call__(self, x):
            x = torch.tensor(x)
            return Normal(x * self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Fit the model
    x = np.random.randn(100).astype("float32")
    y = -x + 1
    my_model.fit(x, y, batch_size=5, epochs=3)

    # Shouldn't be training
    assert my_model._is_training is False

    # Should be able to set learning rate
    lr = my_model._learning_rate
    my_model.set_learning_rate(lr + 1.0)
    assert lr != my_model._learning_rate

    # but error w/ wrong type
    with pytest.raises(TypeError):
        my_model.set_learning_rate("asdf")

    # predictive samples
    samples = my_model.predictive_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # aleatoric samples
    samples = my_model.aleatoric_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # epistemic samples
    samples = my_model.epistemic_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # predict
    samples = my_model.predict(x[:30])
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 30
    with pytest.raises(ValueError):
        samples = my_model.predict(x[:30], method="asdf")

    # metric
    metric = my_model.metric("mae", x[:30], y[:30])
    assert isinstance(metric, np.floating)
    metric = my_model.metric("mse", x[:30], y[:30])
    assert isinstance(metric, np.floating)
    assert metric >= 0

    # posterior_mean w/ no args should return all params
    val = my_model.posterior_mean()
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)

    # posterior_mean w/ str should return value of that param
    val = my_model.posterior_mean("Weight")
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1

    # posterior_mean w/ list of params should return only those params
    val = my_model.posterior_mean(["Weight", "Std"])
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)

    # posterior_sample w/ no args should return all params
    val = my_model.posterior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    assert all(val[v].shape[1] == 1 for v in val)

    # posterior_sample w/ str should return sample of that param
    val = my_model.posterior_sample("Weight", n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 2
    assert val.shape[0] == 20
    assert val.shape[1] == 1

    # posterior_sample w/ list of params should return only those params
    val = my_model.posterior_sample(["Weight", "Std"], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    assert all(val[v].shape[1] == 1 for v in val)

    # posterior_ci should return confidence intervals of all params by def
    val = my_model.posterior_ci(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 1 for v in val)
    assert all(val[v][1].ndim == 1 for v in val)
    assert all(val[v][0].shape[0] == 1 for v in val)
    assert all(val[v][1].shape[0] == 1 for v in val)

    # posterior_ci should return ci of only 1 if passed str
    val = my_model.posterior_ci("Weight", n=20)
    assert isinstance(val, tuple)
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)

    # posterior_ci should return specified cis if passed list of params
    val = my_model.posterior_ci(["Weight", "Std"], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 1 for v in val)
    assert all(val[v][1].ndim == 1 for v in val)
    assert all(val[v][0].shape[0] == 1 for v in val)
    assert all(val[v][1].shape[0] == 1 for v in val)

    # prior_sample w/ no args should return all params
    val = my_model.prior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)

    # prior_sample w/ str should return sample of that param
    val = my_model.prior_sample("Weight", n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    assert val.shape[0] == 20

    # prior_sample w/ list of params should return only those params
    val = my_model.prior_sample(["Weight", "Std"], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)

    # log_prob should return log prob of each sample by default
    probs = my_model.log_prob(x[:30], y[:30])
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 30

    # log_prob should return sum if individually = False
    s_prob = my_model.log_prob(x[:30], y[:30], individually=False)
    assert isinstance(s_prob, np.floating)
    assert s_prob == np.sum(probs)

    # log_prob should return samples w/ distribution = True
    probs = my_model.log_prob(x[:30], y[:30], n=10, distribution=True)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 30
    assert probs.shape[1] == 10

    # log_prob should return samples w/ distribution = True
    probs = my_model.log_prob(
        x[:30], y[:30], n=10, distribution=True, individually=False
    )
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 10

    # prob should return prob of each sample by default
    probs = my_model.prob(x[:30], y[:30])
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 30
    assert np.all(probs >= 0)

    # prob should return sum if individually = False
    s_prob = my_model.prob(x[:30], y[:30], individually=False)
    assert isinstance(s_prob, np.floating)

    # prob should return samples w/ distribution = True
    probs = my_model.prob(x[:30], y[:30], n=10, distribution=True)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 30
    assert probs.shape[1] == 10
    assert np.all(probs >= 0)

    # prob should return samples w/ distribution = True
    probs = my_model.prob(
        x[:30], y[:30], n=10, distribution=True, individually=False
    )
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 10
    assert np.all(probs >= 0)

    # summary method should run
    my_model.summary()


def test_Model_force_eager():
    """Tests fitting probflow.model.Model forcing eager=True"""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter(name="Weight")
            self.bias = Parameter(name="Bias")
            self.std = ScaleParameter(name="Std")

        def __call__(self, x):
            x = torch.tensor(x)
            return Normal(x * self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Fit it with tracing off
    x = np.random.randn(100).astype("float32")
    y = -x + 1
    my_model.fit(x, y, batch_size=50, epochs=2, eager=True)


def test_Model_with_dataframe():
    """Tests fitting probflow.model.Model w/ DataFrame and eager=False"""

    import pandas as pd

    class MyModel(Model):
        def __init__(self, cols):
            self.cols = cols
            self.weight = Parameter([len(cols), 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")
            self.std = ScaleParameter([1, 1], name="Std")

        def __call__(self, x):
            x = torch.tensor(x[self.cols].values)
            return Normal(x @ self.weight() + self.bias(), self.std())

    # Data
    N = 256
    D = 3
    cols = ["feature1", "feature2", "feature3"]
    x_np = np.random.randn(N, D).astype("float32")
    w = np.random.randn(D, 1).astype("float32")
    y = x_np @ w + 0.1 * np.random.randn(N, 1).astype("float32")
    x_df = pd.DataFrame(x_np, columns=cols)
    y_s = pd.Series(y[:, 0])

    # Instantiate the model
    my_model = MyModel(cols)

    # Fitting should work w/ DataFrame b/c it falls back on eager
    my_model.fit(x_df, y_s, epochs=2)

    # And should still work with eager execution when set
    my_model.fit(x_df, y_s, epochs=2, eager=True)


def test_Model_ArrayDataGenerators():
    """Tests the probflow.models.Model sampling/predictive methods when
    passed ArrayDataGenerators"""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter(name="Weight")
            self.bias = Parameter(name="Bias")
            self.std = ScaleParameter(name="Std")

        def __call__(self, x):
            x = torch.tensor(x)
            return Normal(x * self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Make a ArrayDataGenerator
    x = np.random.randn(100).astype("float32")
    y = -x + 1
    data = ArrayDataGenerator(x, y, batch_size=5)

    # Fit the model
    my_model.fit(data, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # aleatoric samples
    samples = my_model.aleatoric_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # epistemic samples
    samples = my_model.epistemic_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # predict
    samples = my_model.predict(data)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 100

    # metric
    metric = my_model.metric("mae", data)
    assert isinstance(metric, np.floating)
    metric = my_model.metric("mse", data)
    assert isinstance(metric, np.floating)
    assert metric >= 0


def test_Model_1D():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter([5, 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")
            self.std = ScaleParameter([1, 1], name="Std")

        def __call__(self, x):
            x = torch.tensor(x)
            return Normal(x @ self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1

    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # aleatoric samples
    samples = my_model.aleatoric_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # epistemic samples
    samples = my_model.epistemic_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # predict
    samples = my_model.predict(x[:30, :])
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 30
    assert samples.shape[1] == 1

    # metric
    metric = my_model.metric("mse", x[:30, :], y[:30, :])
    assert isinstance(metric, np.floating)
    metric = my_model.metric("mae", x[:30, :], y[:30, :])
    assert isinstance(metric, np.floating)
    assert metric >= 0

    # posterior_mean w/ no args should return all params
    val = my_model.posterior_mean()
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert val["Weight"].shape[0] == 5
    assert val["Weight"].shape[1] == 1
    assert val["Bias"].shape[0] == 1
    assert val["Bias"].shape[1] == 1
    assert val["Std"].shape[0] == 1
    assert val["Std"].shape[1] == 1

    # posterior_mean w/ str should return value of that param
    val = my_model.posterior_mean("Weight")
    assert isinstance(val, np.ndarray)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 1

    # posterior_mean w/ list of params should return only those params
    val = my_model.posterior_mean(["Weight", "Std"])
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert val["Weight"].shape[0] == 5
    assert val["Weight"].shape[1] == 1
    assert val["Std"].shape[0] == 1
    assert val["Std"].shape[1] == 1

    # posterior_sample w/ no args should return all params
    val = my_model.posterior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 3 for v in val)
    assert val["Weight"].shape[0] == 20
    assert val["Weight"].shape[1] == 5
    assert val["Weight"].shape[2] == 1
    assert val["Bias"].shape[0] == 20
    assert val["Bias"].shape[1] == 1
    assert val["Bias"].shape[2] == 1
    assert val["Std"].shape[0] == 20
    assert val["Std"].shape[1] == 1
    assert val["Std"].shape[2] == 1

    # posterior_sample w/ str should return sample of that param
    val = my_model.posterior_sample("Weight", n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 3
    assert val.shape[0] == 20
    assert val.shape[1] == 5
    assert val.shape[2] == 1

    # posterior_sample w/ list of params should return only those params
    val = my_model.posterior_sample(["Weight", "Std"], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 3 for v in val)
    assert val["Weight"].shape[0] == 20
    assert val["Weight"].shape[1] == 5
    assert val["Weight"].shape[2] == 1
    assert val["Std"].shape[0] == 20
    assert val["Std"].shape[1] == 1
    assert val["Std"].shape[2] == 1

    # posterior_ci should return confidence intervals of all params by def
    val = my_model.posterior_ci(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 2 for v in val)
    assert all(val[v][1].ndim == 2 for v in val)
    for i in range(1):
        assert val["Weight"][i].shape[0] == 5
        assert val["Weight"][i].shape[1] == 1
        assert val["Bias"][i].shape[0] == 1
        assert val["Bias"][i].shape[1] == 1
        assert val["Std"][i].shape[0] == 1
        assert val["Std"][i].shape[1] == 1

    # posterior_ci should return ci of only 1 if passed str
    val = my_model.posterior_ci("Weight", n=20)
    assert isinstance(val, tuple)
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)

    # posterior_ci should return specified cis if passed list of params
    val = my_model.posterior_ci(["Weight", "Std"], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 2 for v in val)
    assert all(val[v][1].ndim == 2 for v in val)
    for i in range(1):
        assert val["Weight"][i].shape[0] == 5
        assert val["Weight"][i].shape[1] == 1
        assert val["Std"][i].shape[0] == 1
        assert val["Std"][i].shape[1] == 1

    # prior_sample w/ no args should return all params
    val = my_model.prior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert "Weight" in val
    assert "Bias" in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)

    # prior_sample w/ str should return sample of that param
    val = my_model.prior_sample("Weight", n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    assert val.shape[0] == 20

    # prior_sample w/ list of params should return only those params
    val = my_model.prior_sample(["Weight", "Std"], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert "Weight" in val
    assert "Bias" not in val
    assert "Std" in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)


def test_generative_Model():
    """Tests the probflow.models.Model w/ a generative model (only x)"""

    class MyModel(Model):
        def __init__(self):
            self.mean = Parameter([1], name="Mean")
            self.std = ScaleParameter([1], name="Std")

        def __call__(self):
            return Normal(self.mean(), self.std())

    # Instantiate the model
    model = MyModel()

    # Data
    X = np.random.randn(100, 1).astype("float32")

    # Fit the model
    model.fit(X, batch_size=10, epochs=3)

    # predictive samples
    samples = model.predictive_sample(n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 1

    # log_prob
    y = np.random.randn(10, 1).astype("float32")
    probs = model.log_prob(y)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 10
    assert probs.shape[1] == 1

    probs = model.log_prob(y, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 1

    probs = model.log_prob(y, distribution=True, n=11)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 3
    assert probs.shape[0] == 10
    assert probs.shape[1] == 1
    assert probs.shape[2] == 11

    probs = model.log_prob(y, distribution=True, n=11, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 1
    assert probs.shape[1] == 11


def test_Model_nesting():
    """Tests Model when it contains Modules and sub-modules"""

    class MyModule(Module):
        def __init__(self):
            self.weight = Parameter([5, 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")

        def __call__(self, x):
            return x @ self.weight() + self.bias()

    class MyModel(Model):
        def __init__(self):
            self.module = MyModule()
            self.std = ScaleParameter(
                [1, 1],
                name="Std",
                prior=torch.distributions.gamma.Gamma(1.0, 1.0),
            )

        def __call__(self, x):
            x = torch.tensor(x)
            return Normal(self.module(x), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1

    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # kl loss should be greater for outer model
    assert (
        my_model.kl_loss().detach().numpy()
        > my_model.module.kl_loss().detach().numpy()
    )


def test_ContinuousModel():
    """Tests probflow.models.ContinuousModel"""
    pass
    # TODO


def test_DiscreteModel():
    """Tests probflow.models.DiscreteModel"""
    pass
    # TODO


def test_CategoricalModel():
    """Tests probflow.models.CategoricalModel"""
    pass
    # TODO
