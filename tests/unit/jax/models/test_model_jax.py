import numpy as np

from probflow.distributions import Normal
from probflow.models import Model
from probflow.parameters import (
    Parameter,
    ScaleParameter,
)


def test_Model_0D_jax():
    """Tests the probflow.models.Model abstract base class."""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter(name="Weight")
            self.bias = Parameter(name="Bias")
            self.std = ScaleParameter(name="Std")

        def __call__(self, x):
            # NOTE: testing that we DON'T need to convert x to a tensor explicitly with JAX
            return Normal(x * self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Fit the model
    x = np.random.randn(100).astype("float32")
    y = -x + 1
    my_model.fit(x, y, batch_size=5, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30