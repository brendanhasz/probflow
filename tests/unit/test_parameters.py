"""Tests probflow.parameters modules"""

import pytest

from probflow.parameters import Parameter


def test_parameter_build():
    """Tests probflow.parameters.Parameter._build"""
    v1 = Parameter(name='test_parameter_build')
    v1._build(None)

def test_parameter_ensure_is_built():
    """Tests probflow.parameters.Parameter._ensure_is_built"""
    v1 = Parameter(name='test_parameter_ensure_is_built')
    with pytest.raises(RuntimeError):
        v1._ensure_is_built()
    v1._build(None)
    v1._ensure_is_built()

# TODO: _sample

# TODO: _mean

# TODO: _log_loss

# TODO: _kl_loss

# TODO: posterior
