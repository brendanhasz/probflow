"""Abstract base classes.

The core.base module contains abstract base classes (ABCs) for all of
ProbFlow’s classes.

"""


from abc import ABC



class BaseParameter(ABC):
    """Abstract base class for ProbFlow Parameters"""
    pass



class BaseDistribution(ABC):
    """Abstract base class for ProbFlow Distributions"""
    pass



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
