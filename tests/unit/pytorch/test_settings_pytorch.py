"""Tests the probflow.core.settings module when backend = pytorch"""

import pytest

import torch

import probflow as pf



def test_datatype():
    """Tests get and set_datatype"""

    assert isinstance(pf.get_datatype(), torch.dtype)
    assert pf.get_datatype() == torch.float32

    pf.set_datatype(torch.float64)
    assert isinstance(pf.get_datatype(), torch.dtype)
    assert pf.get_datatype() == torch.float64
    pf.set_datatype(torch.float32)

    with pytest.raises(TypeError):
        pf.set_datatype('lala')
