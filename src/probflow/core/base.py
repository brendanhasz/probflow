"""
The core.base module contains abstract base classes (ABCs) for all of
ProbFlow’s classes.

"""



__all__ = [
    'BaseDistribution',
    'BaseParameter',
    'BaseModule',
    'BaseDataGenerator',
    'BaseCallback',
]



from abc import ABC, abstractmethod
from math import ceil

from probflow.core.settings import get_backend



class BaseDistribution(ABC):
    """Abstract base class for ProbFlow Distributions"""


    @abstractmethod
    def __init__(self, *args):
        """Initialize the distribution"""


    @abstractmethod
    def __call__(self):
        """Get the distribution object from the backend"""


    def prob(self, y):
        """Compute the probability of some data given this distribution"""
        if get_backend() == 'pytorch':
            return self().log_prob(y).exp()
        else:
            return self().prob(y)


    def log_prob(self, y):
        """Compute the log probability of some data given this distribution"""
        return self().log_prob(y)


    def mean(self):
        """Compute the mean of this distribution

        Note that this uses the mode of distributions for which the mean
        is undefined (for example, a categorical distribution)"""
        try:
            return self().mean()
        except NotImplementedError:
            return self().mode()


    def sample(self, n=1):
        """Generate a random sample from this distribution"""
        if get_backend() == 'pytorch':
            if isinstance(n, int) and n == 1:
                return self().rsample()
            else:
                return self().rsample(n)
        else:
            if isinstance(n, int) and n == 1:
                return self().sample()
            else:
                return self().sample(n)



class BaseParameter(ABC):
    """Abstract base class for ProbFlow Parameters"""

    @abstractmethod
    def __init__(self, *args):
        """Initialize the parameter"""


    @abstractmethod
    def __call__(self):
        """Return a sample from or the MAP estimate of this parameter."""


    @abstractmethod
    def kl_loss(self):
        """Compute the sum of the Kullback–Leibler divergences between this
        parameter's priors and its variational posteriors."""


    @abstractmethod
    def posterior_mean(self):
        """Get the mean of the posterior distribution(s)."""


    @abstractmethod
    def posterior_sample(self):
        """Get the mean of the posterior distribution(s)."""


    @abstractmethod
    def prior_sample(self):
        """Get the mean of the posterior distribution(s)."""



class BaseModule(ABC):
    """Abstract base class for ProbFlow Modules"""

    @abstractmethod
    def __init__(self, *args):
        """Initialize the module (abstract method)"""


    @abstractmethod
    def __call__(self):
        """Perform forward pass (abstract method)"""



class BaseDataGenerator(ABC):
    """Abstract base class for ProbFlow DataGenerators"""


    @abstractmethod
    def __init__(self, *args):
        """Initialize the data generator"""


    @property
    @abstractmethod
    def n_samples(self):
        """Number of samples in the dataset"""


    @property
    @abstractmethod
    def batch_size(self):
        """Number of samples per batch"""


    def __len__(self):
        """Number of batches per epoch"""
        return int(ceil(self.n_samples/self.batch_size))


    @abstractmethod
    def __getitem__(self, index):
        """Generate one batch of data"""


    def __iter__(self):
        """Get an iterator over batches"""
        self._batch = -1
        return self

    
    def __next__(self):
        """Get the next batch"""
        self._batch += 1
        if self._batch < len(self):
            return self[self._batch]
        else:
            raise StopIteration()


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""



class BaseCallback(ABC):
    """Abstract base class for ProbFlow Callbacks"""


    # Reference to the model
    model = None


    @abstractmethod
    def __init__(self, *args):
        """Initialize the callback"""


    @abstractmethod
    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""


    @abstractmethod
    def on_train_end(self):
        """Will be called at the end of training"""
