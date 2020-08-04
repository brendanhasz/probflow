from typing import List

import probflow.utils.ops as O
from probflow.distributions import Normal
from probflow.models import ContinuousModel
from probflow.modules import DenseNetwork
from probflow.parameters import ScaleParameter
from probflow.utils.casting import to_tensor


class DenseRegression(ContinuousModel):
    r"""A regression using a multilayer dense neural network

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        dimensionality of the dependent variable (number of dimensions of the
        target).
    heteroscedastic : bool
        Whether to model a change in noise as a function of :math:`\mathbf{x}`
        (if ``heteroscedastic=True``), or not (if ``heteroscedastic=False``,
        the default).
    kwargs
        Additional keyword arguments are passed to :class:`.DenseNetwork`

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        mean
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, d: List[int], heteroscedastic: bool = False, **kwargs):
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.d_out = d[-1]
            d[-1] = 2 * d[-1]
            self.network = DenseNetwork(d, **kwargs)
        else:
            self.network = DenseNetwork(d, **kwargs)
            self.std = ScaleParameter([1, d[-1]], name="std")

    def __call__(self, x):
        x = to_tensor(x)
        if self.heteroscedastic:
            p = self.network(x)
            m_preds = p[..., :, : self.d_out]
            s_preds = O.exp(p[..., :, self.d_out :])
            return Normal(m_preds, s_preds)
        else:
            return Normal(self.network(x), self.std())
