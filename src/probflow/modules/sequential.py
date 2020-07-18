from typing import Callable, List

from .module import Module


class Sequential(Module):
    """Apply a series of modules or functions sequentially.

    TODO

    Parameters
    ----------
    steps : list of |Modules| or callables
        Steps to apply
    name : str
        Name of this module
    """

    def __init__(self, steps: List[Callable], name: str = "Sequential"):
        self.steps = steps  # store the list of steps

    def __call__(self, x):
        """Perform the forward pass"""
        for step in self.steps:
            x = step(x)
        return x
