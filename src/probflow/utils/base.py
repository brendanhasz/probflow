"""
The utils.base module contains abstract base classes (ABCs) for all of
ProbFlow’s classes.

"""

__all__ = [
    "BaseCallback",
    "BaseDataGenerator",
    "BaseDistribution",
    "BaseModule",
    "BaseParameter",
]


from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from probflow.utils.casting import to_tensor
from probflow.utils.settings import get_backend
from probflow.utils.typing import (
    BackendDistribution,
    BackendTensor,
    BackendVariable,
    ScalarLike,
)


class BaseDistribution(ABC):
    """Abstract base class for ProbFlow Distributions"""

    @abstractmethod
    def __init__(self, *args):
        """Initialize the distribution"""

    def __call__(self):
        """Get the distribution object from the backend"""

    def __getitem__(self, key):
        """Get a parameter, or if a probflow.Parameter, get a sample"""
        param = getattr(self, key)
        if callable(param):
            return param()
        else:
            return param

    def prob(self, y):
        """Compute the probability of some data given this distribution"""
        if get_backend() == "pytorch":
            return self().log_prob(to_tensor(y)).exp()
        else:
            return self().prob(to_tensor(y))

    def log_prob(self, y):
        """Compute the log probability of some data given this distribution"""
        return self().log_prob(to_tensor(y))

    def cdf(self, y):
        """Cumulative probability of some data along this distribution"""
        return self().cdf(to_tensor(y))

    def mean(self):
        """Compute the mean of this distribution

        Note that this uses the mode of distributions for which the mean
        is undefined (for example, a categorical distribution)"""
        if get_backend() == "pytorch":
            return self().mean
        else:
            try:
                return self().mean()
            except NotImplementedError:
                return self().mode()

    def mode(self):
        """Compute the mode of this distribution"""
        if get_backend() == "pytorch":
            return self().mode
        else:
            return self().mode()

    def sample(self, n=1):
        """Generate a random sample from this distribution"""
        if get_backend() == "pytorch":
            try:
                if isinstance(n, int) and n == 1:
                    return self().rsample()
                elif isinstance(n, int):
                    return self().rsample([n])
                else:
                    return self().rsample(n)
            except NotImplementedError:
                if isinstance(n, int) and n == 1:
                    return self().sample()
                elif isinstance(n, int):
                    return self().sample([n])
                else:
                    return self().sample(n)
        else:
            if isinstance(n, int) and n == 1:
                return self().sample()
            else:
                return self().sample(n)


class BaseParameter(ABC):
    """Abstract base class for ProbFlow Parameters"""

    name: str

    @abstractmethod
    def __init__(self, *args):
        """Initialize the parameter"""

    @abstractmethod
    def __call__(self) -> BackendTensor:
        """Return a sample from or the MAP estimate of this parameter."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Get the number of independent parameters"""

    @property
    @abstractmethod
    def n_variables(self) -> int:
        """Get the number of underlying variables"""

    @property
    @abstractmethod
    def trainable_variables(self) -> list[BackendVariable]:
        """Get a list of trainable variables from the backend"""

    @property
    @abstractmethod
    def variables(self) -> dict[str, BackendTensor]:
        """Variables after applying their respective transformations"""

    @property
    @abstractmethod
    def posterior(self) -> BaseDistribution:
        """This Parameter's variational posterior distribution"""

    @abstractmethod
    def kl_loss(self) -> ScalarLike:
        """Compute the sum of the Kullback–Leibler divergences between this
        parameter's priors and its variational posteriors."""

    @abstractmethod
    def bayesian_update(self) -> None:
        """Update priors to match the current posterior."""

    @abstractmethod
    def posterior_mean(self) -> np.ndarray:
        """Get the mean of the posterior distribution(s)."""

    @abstractmethod
    def posterior_sample(self, n: int = 1) -> np.ndarray:
        """Sample from the posterior distribution."""

    @abstractmethod
    def prior_sample(self, n: int = 1) -> np.ndarray:
        """Sample from the prior distribution."""


class BaseModule(ABC):
    """Abstract base class for ProbFlow Modules"""

    _kl_losses: list[ScalarLike]

    @abstractmethod
    def __init__(self, *args):
        """Initialize the module (abstract method)"""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> BackendTensor:
        """Perform forward pass, returning a tensor (abstract method)"""

    @property
    @abstractmethod
    def parameters(self) -> list[BaseParameter]:
        """A list of |Parameters| in this |Module| and its sub-Modules."""

    @property
    @abstractmethod
    def modules(self) -> list["BaseModule"]:
        """A list of sub-Modules in this |Module|, including itself."""

    @property
    @abstractmethod
    def trainable_variables(self) -> list[BackendVariable]:
        """A list of trainable backend variables within this |Module|"""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Get the number of independent parameters of this module"""

    @property
    @abstractmethod
    def n_variables(self) -> int:
        """Get the number of underlying variables in this module"""

    @abstractmethod
    def bayesian_update(self) -> None:
        """Perform a Bayesian update of all |Parameters| in this module.  Sets
        the prior to the current variational posterior for all parameters.
        """

    @abstractmethod
    def kl_loss(self) -> ScalarLike:
        """Compute the sum of the Kullback-Leibler divergences between
        priors and their variational posteriors for all |Parameters| in this
        |Module| and its sub-Modules."""

    @abstractmethod
    def kl_loss_batch(self) -> ScalarLike:
        """Compute the sum of additional Kullback-Leibler divergences due to
        data in this batch"""

    @abstractmethod
    def reset_kl_loss(self) -> None:
        """Reset additional loss due to KL divergences"""

    @abstractmethod
    def add_kl_loss(self, loss: ScalarLike) -> None:
        """Add additional loss due to KL divergences."""

    @abstractmethod
    def add_kl_loss_between(
        self,
        d1: BaseDistribution | BackendDistribution,
        d2: BaseDistribution | BackendDistribution,
    ) -> None:
        """Add additional loss due to KL divergences between two distributions."""

    @abstractmethod
    def dumps(self) -> str:
        """Serialize module object to bytes."""

    @abstractmethod
    def save(self, filename: str | Path) -> None:
        """Save module object to file."""


class BaseModel(ABC):
    """Abstract base class for ProbFlow Models"""

    @abstractmethod
    def __init__(self, *args):
        """Initialize the model (abstract method)"""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> BaseDistribution:
        """Perform forward pass, returning a distribution (abstract method)"""


class BaseDataGenerator(ABC):
    """Abstract base class for ProbFlow DataGenerators"""

    @abstractmethod
    def __init__(self, *args):
        """Initialize the data generator"""

    def on_epoch_start(self):
        """Will be called at the start of each training epoch"""

    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""

    @property
    @abstractmethod
    def n_samples(self):
        """Number of samples in the dataset"""

    @property
    @abstractmethod
    def batch_size(self):
        """Number of samples to generate each minibatch"""

    def __len__(self):
        """Number of batches per epoch"""
        return ceil(self.n_samples / self.batch_size)

    @abstractmethod
    def __getitem__(self, index):
        """Generate one batch of data"""

    @abstractmethod
    def __iter__(self):
        """Get an iterator over batches"""

    @abstractmethod
    def __next__(self):
        """Get the next batch"""


class BaseCallback(ABC):
    """Abstract base class for ProbFlow Callbacks"""

    # Reference to the model
    model: Any = None

    @abstractmethod
    def __init__(self, *args):
        """Initialize the callback"""

    @abstractmethod
    def on_train_start(self):
        """Will be called at the start of training"""

    @abstractmethod
    def on_epoch_start(self):
        """Will be called at the start of each training epoch"""

    @abstractmethod
    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""

    @abstractmethod
    def on_train_end(self):
        """Will be called at the end of training"""
