"""Abstract base classes.

The core.base module contains abstract base classes (ABCs) for all of
ProbFlow’s classes.

"""


from abc import ABC, abstractmethod

from probflow.core.settings import get_backend


# Import the relevant backend
if get_backend() == 'pytorch':
    import torch
    tod = torch.distributions
else:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions



class BaseParameter(ABC):
    """Abstract base class for ProbFlow Parameters"""
    pass



class BaseDistribution(ABC):
    """Abstract base class for ProbFlow Distributions"""


    @abstractmethod
    def __init__(self, *args):
        pass


    @abstractmethod
    def __call__(self):
        """Get the distribution object from the backend"""
        pass


    def prob(self, y):
        """Compute the probability of some data given this distribution"""
        if get_backend() == 'pytorch':
            return self.__call__().log_prob(y).exp()
        else:
            return self.__call__().prob(y)


    def log_prob(self, y):
        """Compute the log probability of some data given this distribution"""
        return self.__call__().log_prob(y)


    def mean(self):
        """Compute the mean of this distribution"""
        return self.__call__().mean()


    def sample(self, n=1):
        """Compute the probability of some data given this distribution"""
        if get_backend() == 'pytorch':
            if isinstance(n, int) and n > 1:
                return self.__call__().rsample()
            else:
                return self.__call__().rsample(n)
        else:
            if isinstance(n, int) and n > 1:
                return self.__call__().sample()
            else:
                return self.__call__().sample(n)



class BaseModule(ABC):
    """Abstract base class for ProbFlow Modules"""
    pass



class BaseModel(ABC):
    """Abstract base class for ProbFlow Models"""
    pass



class BaseDataGenerator(ABC):
    """Abstract base class for ProbFlow DataGenerators"""
    pass



class BaseCallback(ABC):
    """Abstract base class for ProbFlow Callbacks"""
    pass
