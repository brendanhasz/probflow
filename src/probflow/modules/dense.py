import probflow.utils.ops as O
from probflow.parameters import Parameter
from probflow.utils.settings import get_backend, get_flipout

from .module import Module


class Dense(Module):
    """Dense neural network layer.

    TODO

    Parameters
    ----------
    d_in : int
        Number of input dimensions.
    d_out : int
        Number of output dimensions (number of "units").
    name : str
        Name of this layer
    """

    def __init__(self, d_in: int, d_out: int = 1, name: str = "Dense"):

        # Check types
        if d_in < 1:
            raise ValueError("d_in must be >0")
        if d_out < 1:
            raise ValueError("d_out must be >0")

        # Create the parameters
        self.d_in = d_in
        self.d_out = d_out
        self.weights = Parameter(shape=[d_in, d_out], name=name + "_weights")
        self.bias = Parameter(shape=[1, d_out], name=name + "_bias")

    def __call__(self, x):
        """Perform the forward pass"""

        # Using the Flipout estimator
        if get_flipout():

            # With PyTorch
            if get_backend() == "pytorch":
                raise NotImplementedError

            # With Tensorflow
            else:

                import tensorflow as tf
                import tensorflow_probability as tfp

                # Flipout-estimated weight samples
                s = tfp.python.math.random_rademacher(tf.shape(x))
                r = tfp.python.math.random_rademacher([x.shape[0], self.d_out])
                norm_samples = tf.random.normal([self.d_in, self.d_out])
                w_samples = self.weights.variables["scale"] * norm_samples
                w_noise = r * ((x * s) @ w_samples)
                w_outputs = x @ self.weights.variables["loc"] + w_noise

                # Flipout-estimated bias samples
                r = tfp.python.math.random_rademacher([x.shape[0], self.d_out])
                norm_samples = tf.random.normal([self.d_out])
                b_samples = self.bias.variables["scale"] * norm_samples
                b_outputs = self.bias.variables["loc"] + r * b_samples

                return w_outputs + b_outputs

        # Without Flipout
        else:
            return x @ self.weights() + self.bias()
