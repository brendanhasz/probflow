"""
The :mod:`.distributions` module contains classes to instantiate probability
distributions, which describe the likelihood of either a parameter or a
datapoint taking any given value.  Distribution objects are used to represent
both the predicted probability distribution of the data, and also the
parameters' posteriors and priors.

Continuous Distributions
------------------------

* :class:`.Deterministic`
* :class:`.Normal`
* :class:`.MultivariateNormal`
* :class:`.StudentT`
* :class:`.Cauchy`
* :class:`.Gamma`
* :class:`.InverseGamma`

Discrete Distributions
----------------------

* :class:`.Bernoulli`
* :class:`.Categorical`
* :class:`.OneHotCategorical`
* :class:`.Poisson`
* :class:`.Dirichlet`

Other
-----

* :class:`.Mixture`
* :class:`.HiddenMarkovModel`

----------

"""


__all__ = [
    "Bernoulli",
    "Categorical",
    "Cauchy",
    "Deterministic",
    "Dirichlet",
    "Gamma",
    "HiddenMarkovModel",
    "InverseGamma",
    "Mixture",
    "MultivariateNormal",
    "Normal",
    "OneHotCategorical",
    "Poisson",
    "StudentT",
]


from .bernoulli import Bernoulli
from .categorical import Categorical
from .cauchy import Cauchy
from .deterministic import Deterministic
from .dirichlet import Dirichlet
from .gamma import Gamma
from .hidden_markov_model import HiddenMarkovModel
from .inverse_gamma import InverseGamma
from .mixture import Mixture
from .multivariate_normal import MultivariateNormal
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .poisson import Poisson
from .student_t import StudentT
