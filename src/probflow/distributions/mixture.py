from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Mixture(BaseDistribution):
    r"""A mixture distribution.

    TODO

    TODO: example image of the distribution w/ 2 gaussians


    Parameters
    ----------
    concentration : |ndarray|, or Tensor
        Concentration parameter of the Dirichlet distribution (:math:`\alpha`).
    """

    def __init__(self, distributions, logits=None, probs=None):

        # Check input
        # TODO: distributions should be a pf, tf, or pt distribution
        if logits is None and probs is None:
            raise ValueError("must pass either logits or probs")
        if probs is not None:
            ensure_tensor_like(probs, "probs")
        if logits is not None:
            ensure_tensor_like(logits, "logits")

        # Store args
        self.distributions = distributions
        self.logits = logits
        self.probs = probs

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            # import torch.distributions as tod
            raise NotImplementedError
        else:
            import tensorflow as tf
            from tensorflow_probability import distributions as tfd

            # Convert to tensorflow distributions if probflow distributions
            if isinstance(self.distributions, BaseDistribution):
                self.distributions = self.distributions()

            # Broadcast probs/logits
            shape = self.distributions.batch_shape
            args = {"logits": None, "probs": None}
            if self.logits is not None:
                args["logits"] = tf.broadcast_to(self["logits"], shape)
            else:
                args["probs"] = tf.broadcast_to(self["probs"], shape)

            # Return TFP distribution object
            return tfd.MixtureSameFamily(
                tfd.Categorical(**args), self.distributions
            )
