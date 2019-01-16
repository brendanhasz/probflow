"""Tests probflow.variables modules"""

import pytest

from probflow.variables import Variable


def test_variable_build():
    """Tests probflow.variables.Variable._build"""
    v1 = Variable(name='test_variable_build')
    v1._build(None)

def test_variable_ensure_is_built():
    """Tests probflow.variables.Variable._ensure_is_built"""
    v1 = Variable(name='test_variable_ensure_is_built')
    with pytest.raises(RuntimeError):
        v1._ensure_is_built()
    v1._build(None)
    v1._ensure_is_built()

# TODO: _sample

# TODO: _mean

# TODO: _log_loss

# TODO: _kl_loss

# TODO: posterior
