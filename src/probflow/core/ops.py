"""Backend-specific operations

The core.ops module contains operations which run using the current backend.

"""


__all__ = [
    'kl_divergence',
]



from probflow.core.settings import get_backend


# Import the relevant backend
if get_backend() == 'pytorch':
    import torch
    tod = torch.distributions
else:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions



def kl_divergence(P, Q):
    """Compute the Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    P : |Distribution|
        The first distribution
    Q : |Distribution|
        The second distribution

    Returns
    -------
    kld : Tensor
        The Kullback–Leibler divergence between P and Q (KL(P || Q))
    """
    if get_backend() == 'pytorch':
        return tod.kl.kl_divergence(P(), Q())
    else:
        return tfd.kl_divergence(P(), Q())


# TODO: all other ops used by probflow internals


def exp(val):
    """The natural exponent."""
    if get_backend() == 'pytorch':
        return torch.exp(val)
    else:
        return tf.exp(val)



def relu(val):
    """Linear rectification."""
    if get_backend() == 'pytorch':
        return torch.exp(val)
    else:
        return tf.exp(val)
