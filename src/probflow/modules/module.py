"""Abstract base class for Modules."""

from pathlib import Path
from typing import Any

import probflow.utils.ops as O
from probflow.utils.base import BaseDistribution, BaseModule, BaseParameter
from probflow.utils.io import dump, dumps
from probflow.utils.typing import (
    BackendDistribution,
    BackendVariable,
    ScalarLike,
)


class Module(BaseModule):
    r"""Abstract base class for Modules.

    TODO

    """

    def _params(self, obj: Any) -> list[BaseParameter]:
        """Recursively search for |Parameters| contained within an object."""
        if isinstance(obj, BaseParameter):
            return [obj]
        elif isinstance(obj, BaseModule):
            return obj.parameters
        elif isinstance(obj, list):
            return self._list_params(obj)
        elif isinstance(obj, dict):
            return self._dict_params(obj)
        else:
            return []

    def _list_params(self, the_list: list) -> list[BaseParameter]:
        """Recursively search for |Parameters| contained in a list."""
        return [p for e in the_list for p in self._params(e)]

    def _dict_params(self, the_dict: dict) -> list[BaseParameter]:
        """Recursively search for |Parameters| contained in a dict."""
        return [p for _, e in the_dict.items() for p in self._params(e)]

    @property
    def parameters(self) -> list[BaseParameter]:
        """A list of |Parameters| in this |Module| and its sub-Modules."""
        return [p for _, a in vars(self).items() for p in self._params(a)]

    @property
    def modules(self) -> list[BaseModule]:
        """A list of sub-Modules in this |Module|, including itself."""
        return [
            m
            for a in vars(self).values()
            if isinstance(a, BaseModule)
            for m in a.modules
        ] + [self]

    @property
    def trainable_variables(self) -> list[BackendVariable]:
        """A list of trainable backend variables within this |Module|."""
        return [v for p in self.parameters for v in p.trainable_variables]
        # TODO: look for variables NOT in parameters too
        # so users can mix-n-match tf.Variables and pf.Parameters in modules

    @property
    def n_parameters(self) -> int:
        """Get the number of independent parameters of this module."""
        return sum([p.n_parameters for p in self.parameters])

    @property
    def n_variables(self) -> int:
        """Get the number of underlying variables in this module."""
        return sum([p.n_variables for p in self.parameters])

    def bayesian_update(self) -> None:
        """Perform a Bayesian update of all |Parameters| in this module.  Sets
        the prior to the current variational posterior for all parameters.
        """
        for p in self.parameters:
            p.bayesian_update()

    def kl_loss(self) -> ScalarLike:
        """Compute the sum of the Kullback-Leibler divergences between
        priors and their variational posteriors for all |Parameters| in this
        |Module| and its sub-Modules.
        """
        return sum([p.kl_loss() for p in self.parameters])

    def kl_loss_batch(self) -> ScalarLike:
        """Compute the sum of additional Kullback-Leibler divergences due to
        data in this batch.
        """
        return sum([e for m in self.modules for e in m._kl_losses])

    def reset_kl_loss(self) -> None:
        """Reset additional loss due to KL divergences."""
        for m in self.modules:
            m._kl_losses = []

    def add_kl_loss(self, loss: ScalarLike) -> None:
        """Add additional loss due to KL divergences.

        Parameters
        ----------
        loss : TensorLike
            The loss to add to the current model's KL loss.
        """
        self._kl_losses += [O.sum(loss, axis=None)]

    def add_kl_loss_between(
        self,
        d1: BaseDistribution | BackendDistribution,
        d2: BaseDistribution | BackendDistribution,
    ) -> None:
        """Add additional loss due to KL divergences between two distributions.

        Parameters
        ----------
        d1 : |BaseDistribution| or backend distribution
            The first distribution.
        d2 : |BaseDistribution| or backend distribution
            The second distribution.
        """
        self._kl_losses += [O.sum(O.kl_divergence(d1, d2), axis=None)]

    def dumps(self) -> str:
        """Serialize module object to bytes."""
        return dumps(self)

    def save(self, filename: str | Path) -> None:
        """Save module object to file.

        Parameters
        ----------
        filename : str
            Filename for file to which to save this object
        """
        dump(self, filename)
