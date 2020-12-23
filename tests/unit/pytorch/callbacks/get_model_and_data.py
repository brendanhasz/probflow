import numpy as np

from probflow.distributions import Normal
from probflow.models import Model
from probflow.parameters import Parameter, ScaleParameter


def get_model_and_data():
    """Gets a simple model and data for testing callbacks"""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter(name="Weight")
            self.bias = Parameter(name="Bias")
            self.std = ScaleParameter(name="Std")

        def __call__(self, x):
            return Normal(x * self.weight() + self.bias(), self.std())

    # Instantiate the model
    model = MyModel()

    # Some data to fit
    x = np.random.randn(100).astype("float32")
    y = -x + 1

    return model, x, y
