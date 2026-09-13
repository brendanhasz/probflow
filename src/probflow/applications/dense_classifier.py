"""A classifier which uses a multilayer dense neural network."""

import probflow.utils.ops as O
from probflow.distributions import Categorical
from probflow.models import CategoricalModel
from probflow.modules import DenseNetwork
from probflow.utils.casting import to_tensor
from probflow.utils.typing import TensorLike


class DenseClassifier(CategoricalModel):
    r"""A classifier which uses a multilayer dense neural network.

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        number of classes of the target.
    activation : callable
        Activation function to apply to the outputs of each layer.
        Note that the activation function will not be applied to the outputs
        of the final layer.
        Default = :math:`\max ( 0, x )`
    kwargs
        Additional keyword arguments are passed to :class:`.DenseNetwork`

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        class probabilities
    """

    def __init__(self, d: list[int], **kwargs):
        """Initialize the DenseClassifier."""
        d[-1] -= (
            1  # Use first class as "pivot" class, so only need to predict k-1 classes
        )
        self.network = DenseNetwork(d, **kwargs)

    def __call__(self, x: TensorLike) -> Categorical:
        """Forward pass of the DenseClassifier."""
        x = to_tensor(x)
        return Categorical(O.insert_col_of(self.network(x), 0))
