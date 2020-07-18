from typing import Dict, List

import probflow.utils.ops as O
from probflow.utils.base import BaseModule, BaseParameter
from probflow.utils.io import dump, dumps


class Module(BaseModule):
    r"""Abstract base class for Modules.

    TODO

    """

    def _params(self, obj):
        """Recursively search for |Parameters| contained within an object"""
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

    def _list_params(self, the_list: List):
        """Recursively search for |Parameters| contained in a list"""
        return [p for e in the_list for p in self._params(e)]

    def _dict_params(self, the_dict: Dict):
        """Recursively search for |Parameters| contained in a dict"""
        return [p for _, e in the_dict.items() for p in self._params(e)]

    @property
    def parameters(self):
        """A list of |Parameters| in this |Module| and its sub-Modules."""
        return [p for _, a in vars(self).items() for p in self._params(a)]

    @property
    def modules(self):
        """A list of sub-Modules in this |Module|, including itself."""
        return [
            m
            for a in vars(self).values()
            if isinstance(a, BaseModule)
            for m in a.modules
        ] + [self]

    @property
    def trainable_variables(self):
        """A list of trainable backend variables within this |Module|"""
        return [v for p in self.parameters for v in p.trainable_variables]
        # TODO: look for variables NOT in parameters too
        # so users can mix-n-match tf.Variables and pf.Parameters in modules

    @property
    def n_parameters(self):
        """Get the number of independent parameters of this module"""
        return sum([p.n_parameters for p in self.parameters])

    @property
    def n_variables(self):
        """Get the number of underlying variables in this module"""
        return sum([p.n_variables for p in self.parameters])

    def kl_loss(self):
        """Compute the sum of the Kullback-Leibler divergences between
        priors and their variational posteriors for all |Parameters| in this
        |Module| and its sub-Modules."""
        return sum([p.kl_loss() for p in self.parameters])

    def kl_loss_batch(self):
        """Compute the sum of additional Kullback-Leibler divergences due to
        data in this batch"""
        return sum([e for m in self.modules for e in m._kl_losses])

    def reset_kl_loss(self):
        """Reset additional loss due to KL divergences"""
        for m in self.modules:
            m._kl_losses = []

    def add_kl_loss(self, loss, d2=None):
        """Add additional loss due to KL divergences."""
        if d2 is None:
            self._kl_losses += [O.sum(loss, axis=None)]
        else:
            self._kl_losses += [O.sum(O.kl_divergence(loss, d2), axis=None)]

    def dumps(self):
        """Serialize module object to bytes"""
        return dumps(self)

    def save(self, filename: str):
        """Save module object to file

        Parameters
        ----------
        filename : str
            Filename for file to which to save this object

        Example
        -------

        .. code-block:: python3

            import numpy as np
            import probflow as pf

            N = 1024
            D = 7
            x = np.random.randn(N, D).astype('float32')
            w = np.random.randn(D, 1).astype('float32')
            y = x@w + np.random.randn(N, 1).astype('float32')

            model = pf.LinearRegression(7)
            model.fit(x, y)

            # Save the model to file
            model.save("my_model.pfm")

            # Load it back in
            model2 = pf.load("my_model.pfm")

        """
        dump(self, filename)
