"""
Parameters are values which characterize the behavior of a model.  When
fitting a model, we want to find the values of the parameters which
best allow the model to explain the data.  However, with Bayesian modeling
we want not only to find the single *best* value for each parameter, but a
probability distribution which describes how likely any given value of
a parameter is to be the best or true value.

Parameters have both priors (probability distributions which describe how
likely we think different values for the parameter are *before* taking into
consideration the current data), and posteriors (probability distributions
which describe how likely we think different values for the parameter are
*after* taking into consideration the current data).  The prior is set
to a specific distribution before fitting the model.  While the *type* of
distribution used for the posterior is set before fitting the model, the
shape of that distribution (the value of the parameters which define the
distribution) is optimized while fitting the model.
See the :doc:`../user_guide/math` section for more info.

The :class:`.Parameter` class can be used to create any probabilistic
parameter.

For convenience, ProbFlow also includes some classes which are special cases
of a :class:`.Parameter`:

* :class:`.ScaleParameter` - standard deviation parameter
* :class:`.CategoricalParameter` - categorical parameter
* :class:`.DirichletParameter` - parameter with a Dirichlet posterior
* :class:`.BoundedParameter` - parameter which is bounded between 0 and 1
* :class:`.PositiveParameter` - parameter which is always greater than 0
* :class:`.DeterministicParameter` - a non-probabilistic parameter
* :class:`.MultivariateNormalParameter` - parameter with a multivariate Normal posterior

See the :doc:`user guide <../user_guide/parameters>` for more information on Parameters.

----------

"""


__all__ = [
    "BoundedParameter",
    "CategoricalParameter",
    "DeterministicParameter",
    "DirichletParameter",
    "MultivariateNormalParameter",
    "Parameter",
    "PositiveParameter",
    "ScaleParameter",
]


from .bounded_parameter import BoundedParameter
from .categorical_parameter import CategoricalParameter
from .deterministic_parameter import DeterministicParameter
from .dirichlet_parameter import DirichletParameter
from .multivariate_normal_parameter import MultivariateNormalParameter
from .parameter import Parameter
from .positive_parameter import PositiveParameter
from .scale_parameter import ScaleParameter
