from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class OneHotCategorical(BaseDistribution):
    r"""The Categorical distribution, parameterized by categories-len vectors.

    TODO: explain

    TODO: example image of the distribution

    TODO: logits vs probs


    Parameters
    ----------
    logits : int, float, |ndarray|, or Tensor
        Logit-transformed category probabilities
    probs : int, float, |ndarray|, or Tensor
    """

    def __init__(self, logits=None, probs=None):

        # Check input
        if logits is None and probs is None:
            raise TypeError("either logits or probs must be specified")
        if logits is None:
            ensure_tensor_like(probs, "probs")
        if probs is None:
            ensure_tensor_like(logits, "logits")

        # Store args
        self.logits = logits
        self.probs = probs

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.one_hot_categorical.OneHotCategorical(
                logits=self["logits"], probs=self["probs"]
            )
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.OneHotCategorical(
                logits=self["logits"], probs=self["probs"]
            )
