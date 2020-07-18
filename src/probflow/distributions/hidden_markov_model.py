from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class HiddenMarkovModel(BaseDistribution):
    r"""A hidden Markov model distribution

    TODO: docs

    .. math::

        p(X_0) \text{initial probability} \\


    TODO: example image of the distribution


    Parameters
    ----------
    initial : |ndarray|, or Tensor
        Concentration parameter of the Dirichlet distribution (:math:`\alpha`).
    """

    def __init__(self, initial, transition, observation, steps):

        # Check input
        ensure_tensor_like(initial, "initial")
        ensure_tensor_like(transition, "transition")
        # observation should be a pf, tf, or pt distribution
        if not isinstance(steps, int):
            raise TypeError("steps must be an int")
        if steps < 1:
            raise ValueError("steps must be >0")

        # Store observation distribution
        if isinstance(observation, BaseDistribution):
            self.observation = observation()  # store backend distribution
        else:
            self.observation = observation

        # Store other args
        self.initial = initial
        self.transition = transition
        self.steps = steps

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            # import torch.distributions as tod
            raise NotImplementedError
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.HiddenMarkovModel(
                initial_distribution=tfd.Categorical(self["initial"]),
                transition_distribution=tfd.Categorical(self["transition"]),
                observation_distribution=self["observation"],
                num_steps=self["steps"],
            )
