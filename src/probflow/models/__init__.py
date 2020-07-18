"""
Models are objects which take Tensor(s) as input, perform some computation
on those Tensor(s), and output probability distributions.

TODO: more...

* :class:`.Model`
* :class:`.ContinuousModel`
* :class:`.DiscreteModel`
* :class:`.CategoricalModel`

----------

"""


__all__ = [
    "Model",
    "ContinuousModel",
    "DiscreteModel",
    "CategoricalModel",
]


from .categorical_model import CategoricalModel
from .continuous_model import ContinuousModel
from .discrete_model import DiscreteModel
from .model import Model
