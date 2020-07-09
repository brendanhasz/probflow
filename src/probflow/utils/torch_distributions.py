"""Torch backend distributions"""


def get_TorchDeterministic():

    from numbers import Number

    import torch
    from torch.distributions import constraints
    from torch.distributions.distribution import Distribution
    from torch.distributions.utils import broadcast_all

    class TorchDeterministic(torch.distributions.distribution.Distribution):
        """Deterministic distribution for PyTorch"""

        arg_constraints = {"loc": constraints.real}
        support = constraints.real
        has_rsample = True

        @property
        def mean(self):
            return self.loc

        @property
        def stddev(self):
            return 0.0

        @property
        def variance(self):
            return 0.0

        def __init__(self, loc, validate_args=None):
            self.loc = broadcast_all(loc)[0]

            if isinstance(loc, Number):
                batch_shape = torch.Size()
            else:
                batch_shape = self.loc.size()
            super(TorchDeterministic, self).__init__(
                batch_shape, validate_args=validate_args
            )

        def expand(self, batch_shape, _instance=None):
            new = self._get_checked_instance(TorchDeterministic, _instance)
            batch_shape = torch.Size(batch_shape)
            new.loc = self.loc.expand(batch_shape)
            super(TorchDeterministic, new).__init__(
                batch_shape, validate_args=False
            )
            new._validate_args = self._validate_args
            return new

        def rsample(self, sample_shape=torch.Size()):
            shape = self._extended_shape(sample_shape)
            ones = torch.ones(
                shape, dtype=self.loc.dtype, device=self.loc.device
            )
            return self.loc * ones

        def log_prob(self, value):
            if self._validate_args:
                self._validate_sample(value)
            return torch.log(value.eq(self.loc).type_as(self.loc))

        def cdf(self, value):
            if self._validate_args:
                self._validate_sample(value)
            result = value.gt(self.loc).type_as(self.loc)
            return result.clamp(min=0, max=1)

        def icdf(self, value):
            if self._validate_args:
                self._validate_sample(value)
            result = value.lt(self.loc).type_as(self.loc)
            return result

        def entropy(self):
            return torch.log(torch.zeros([1]))

    return TorchDeterministic
