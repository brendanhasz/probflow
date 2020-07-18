from typing import List

import probflow.utils.ops as O
from probflow.distributions import Categorical
from probflow.models import CategoricalModel
from probflow.modules import DenseNetwork
from probflow.utils.casting import to_tensor


class DenseClassifier(CategoricalModel):
    r"""A classifier which uses a multilayer dense neural network

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        number of classes of the target.
    kwargs
        Additional keyword arguments are passed to :class:`.DenseNetwork`

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        class probabilities
    """

    def __init__(self, d: List[int], **kwargs):
        d[-1] -= 1
        self.network = DenseNetwork(d, **kwargs)

    def __call__(self, x):
        x = to_tensor(x)
        return Categorical(O.insert_col_of(self.network(x), 0))
