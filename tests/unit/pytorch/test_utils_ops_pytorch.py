"""Tests the probflow.utils.ops module when backend = pytorch"""


import pytest

import numpy as np
import torch

import probflow as pf
from probflow.utils import ops


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_kl_divergence():
    """Tests kl_divergence"""

    pf.set_backend("pytorch")

    # Divergence between a distribution and itself should be 0
    dist = torch.distributions.normal.Normal(0, 1)
    assert ops.kl_divergence(dist, dist).numpy() == 0.0

    # Divergence between two different distributions should be >0
    d1 = torch.distributions.normal.Normal(0, 1)
    d2 = torch.distributions.normal.Normal(1, 1)
    assert ops.kl_divergence(d1, d2).numpy() > 0.0

    # Divergence between more different distributions should be larger
    d1 = torch.distributions.normal.Normal(0, 1)
    d2 = torch.distributions.normal.Normal(1, 1)
    d3 = torch.distributions.normal.Normal(2, 1)
    assert (
        ops.kl_divergence(d1, d2).numpy() < ops.kl_divergence(d1, d3).numpy()
    )

    # Should auto-convert probflow distibutions
    dist = pf.Normal(0, 1)
    assert ops.kl_divergence(dist, dist).numpy() == 0.0


def test_squeeze():
    """Tests squeeze"""

    x = torch.randn([3, 2, 1])
    xo = ops.squeeze(x)
    assert isinstance(xo, torch.Tensor)
    assert xo.ndim == 2
    assert xo.shape[0] == 3
    assert xo.shape[1] == 2

    x = torch.randn([3, 2])
    xo = ops.squeeze(x)
    assert isinstance(xo, torch.Tensor)
    assert xo.ndim == 2
    assert xo.shape[0] == 3
    assert xo.shape[1] == 2


def test_expand_dims():
    """Tests expand_dims"""

    pf.set_backend("pytorch")

    val = torch.randn(3)
    val = ops.expand_dims(val, 1)
    assert val.ndim == 2
    assert val.shape[0] == 3
    assert val.shape[1] == 1

    val = torch.randn(3)
    val = ops.expand_dims(val, 0)
    assert val.ndim == 2
    assert val.shape[0] == 1
    assert val.shape[1] == 3


def test_ones():
    """Tests ones"""

    pf.set_backend("pytorch")

    # Scalar
    ones = ops.ones([1])
    assert isinstance(ones, torch.Tensor)
    assert ones.ndim == 1
    assert ones.shape[0] == 1
    assert ones.numpy() == 1.0

    # 1D
    ones = ops.ones([5])
    assert isinstance(ones, torch.Tensor)
    assert ones.ndim == 1
    assert ones.shape[0] == 5
    assert all(ones.numpy() == 1.0)

    # 2D
    ones = ops.ones([5, 4])
    assert isinstance(ones, torch.Tensor)
    assert ones.ndim == 2
    assert ones.shape[0] == 5
    assert ones.shape[1] == 4
    assert np.all(ones.numpy() == 1.0)

    # 3D
    ones = ops.ones([5, 4, 3])
    assert isinstance(ones, torch.Tensor)
    assert ones.ndim == 3
    assert ones.shape[0] == 5
    assert ones.shape[1] == 4
    assert ones.shape[2] == 3
    assert np.all(ones.numpy() == 1.0)


def test_zeros():
    """Tests zeros"""

    pf.set_backend("pytorch")

    # Scalar
    zeros = ops.zeros([1])
    assert isinstance(zeros, torch.Tensor)
    assert zeros.ndim == 1
    assert zeros.shape[0] == 1
    assert zeros.numpy() == 0.0

    # 1D
    zeros = ops.zeros([5])
    assert isinstance(zeros, torch.Tensor)
    assert zeros.ndim == 1
    assert zeros.shape[0] == 5
    assert all(zeros.numpy() == 0.0)

    # 2D
    zeros = ops.zeros([5, 4])
    assert isinstance(zeros, torch.Tensor)
    assert zeros.ndim == 2
    assert zeros.shape[0] == 5
    assert zeros.shape[1] == 4
    assert np.all(zeros.numpy() == 0.0)

    # 3D
    zeros = ops.zeros([5, 4, 3])
    assert isinstance(zeros, torch.Tensor)
    assert zeros.ndim == 3
    assert zeros.shape[0] == 5
    assert zeros.shape[1] == 4
    assert zeros.shape[2] == 3
    assert np.all(zeros.numpy() == 0.0)


def test_eye():
    """Tests eye"""

    # Scalar
    eye = ops.eye(4)
    assert isinstance(eye, torch.Tensor)
    assert eye.ndim == 2
    assert eye.shape[0] == 4
    assert eye.shape[1] == 4
    assert eye.numpy()[0, 0] == 1.0
    assert eye.numpy()[0, 1] == 0.0


def test_sum():
    """Tests sum"""

    pf.set_backend("pytorch")

    # Should sum along the last dimension by default
    ones = torch.ones([5, 4, 3])
    val = ops.sum(ones)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 4
    assert np.all(val.numpy() == 3.0)

    # But can change that w/ the axis kwarg
    ones = torch.ones([5, 4, 3])
    val = ops.sum(ones, axis=1)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 3
    assert np.all(val.numpy() == 4.0)

    # Should sum along all dimensions w/ axis=None
    ones = torch.ones([5, 4, 3])
    val = ops.sum(ones, axis=None)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0
    assert val.numpy() == 60

    # Actually test values
    val = ops.sum(torch.Tensor([1.1, 2.0, 3.3]))
    assert is_close(val.numpy(), 6.4)


def test_prod():
    """Tests prod"""

    pf.set_backend("pytorch")

    # Should prod along the last dimension by default
    ones = torch.ones([5, 4, 3])
    val = ops.prod(ones)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 4
    assert np.all(val.numpy() == 1.0)

    # But can change that w/ the axis kwarg
    ones = torch.ones([5, 4, 3])
    val = ops.prod(ones, axis=1)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 3
    assert np.all(val.numpy() == 1.0)

    # Actually test values
    val = ops.prod(torch.Tensor([1.1, 2.0, 3.3]))
    assert is_close(val.numpy(), 7.26)


def test_mean():
    """Tests mean"""

    pf.set_backend("pytorch")

    # Should mean along the last dimension by default
    ones = torch.ones([5, 4, 3])
    val = ops.mean(ones)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 4
    assert np.all(val.numpy() == 1.0)

    # But can change that w/ the axis kwarg
    ones = torch.ones([5, 4, 3])
    val = ops.mean(ones, axis=1)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 3
    assert np.all(val.numpy() == 1.0)

    # Actually test values
    val = ops.mean(torch.Tensor([0.9, 1.9, 2.1, 3.1]))
    assert is_close(val.numpy(), 2.0)


def test_std():
    """Tests std"""

    pf.set_backend("pytorch")

    # Should std along the last dimension by default
    ones = torch.ones([5, 4, 3])
    val = ops.std(ones)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 4
    assert np.all(val.numpy() == 0.0)

    # But can change that w/ the axis kwarg
    ones = torch.ones([5, 4, 3])
    val = ops.std(ones, axis=1)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 3
    assert np.all(val.numpy() == 0.0)

    # Actually test values
    val = ops.std(torch.Tensor([0.9, 1.9, 2.1, 3.1]))
    assert is_close(val.numpy(), np.std([0.9, 1.9, 2.1, 3.1], ddof=1))
    val = ops.std(torch.Tensor([1.0, 2.0, 3.0]))
    assert is_close(val.numpy(), np.std([1.0, 2.0, 3.0], ddof=1))


def _test_elementwise(fn, inputs, outputs):
    """Test elementwise function"""

    # Should be elementwise
    val = fn(torch.randn([5, 4, 3]))
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 3
    assert val.shape[0] == 5
    assert val.shape[1] == 4
    assert val.shape[2] == 3

    # Actually test values
    val = fn(torch.Tensor(inputs))
    for i in range(len(outputs)):
        assert is_close(val.numpy()[i], outputs[i])


def test_round():
    """Tests round"""
    pf.set_backend("pytorch")
    _test_elementwise(
        ops.round, [-0.9, 0.00001, 1.0, 3.14], [-1.0, 0.0, 1.0, 3.0]
    )


def test_abs():
    """Tests abs"""
    pf.set_backend("pytorch")
    _test_elementwise(ops.abs, [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0])


def test_square():
    """Tests square"""
    pf.set_backend("pytorch")
    _test_elementwise(
        ops.square, [-2.0, -1.0, 0.0, 1.0, 3.0], [4.0, 1.0, 0.0, 1.0, 9.0]
    )


def test_sqrt():
    """Tests sqrt"""
    pf.set_backend("pytorch")
    _test_elementwise(ops.sqrt, [0.0, 1.0, 4.0, 100.0], [0.0, 1.0, 2.0, 10.0])


def test_exp():
    """Tests exp"""
    pf.set_backend("pytorch")
    _test_elementwise(
        ops.exp, [-1.0, 0.0, 1.0, 4.0], [np.exp(-1.0), 1.0, np.e, np.exp(4.0)]
    )


def test_relu():
    """Tests relu"""
    pf.set_backend("pytorch")
    _test_elementwise(
        ops.relu, [-1.0, -0.1, 0.0, 0.1, 1.0], [0.0, 0.0, 0.0, 0.1, 1.0]
    )


def test_softplus():
    """Tests softplus"""
    pf.set_backend("pytorch")
    sp = lambda x: np.log(1.0 + np.exp(x))
    _test_elementwise(
        ops.softplus,
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [sp(e) for e in [-2.0, -1.0, 0.0, 1.0, 2.0]],
    )


def test_sigmoid():
    """Tests sigmoid"""
    pf.set_backend("pytorch")
    sm = lambda x: 1.0 / (1.0 + np.exp(-x))
    _test_elementwise(
        ops.sigmoid,
        [-5.0, -1.0, 0.0, 1.0, 5.0],
        [sm(e) for e in [-5.0, -1.0, 0.0, 1.0, 5.0]],
    )


def test_gather():
    """Tests gather"""

    pf.set_backend("pytorch")

    # Should lookup along 1st axis by default
    vals = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    inds = torch.LongTensor([0, 1, 2, 1, 0])
    output = ops.gather(vals, inds)
    assert output.ndim == 2
    assert output.shape[0] == 5
    assert output.shape[1] == 2
    assert output.numpy()[0, 0] == 1.0
    assert output.numpy()[0, 1] == 2.0
    assert output.numpy()[1, 0] == 3.0
    assert output.numpy()[1, 1] == 4.0
    assert output.numpy()[2, 0] == 5.0
    assert output.numpy()[2, 1] == 6.0
    assert output.numpy()[3, 0] == 3.0
    assert output.numpy()[3, 1] == 4.0
    assert output.numpy()[4, 0] == 1.0
    assert output.numpy()[4, 1] == 2.0

    # But can set axis
    inds = torch.LongTensor([1, 0, 1, 0])
    output = ops.gather(vals, inds, axis=1)
    assert output.ndim == 2
    assert output.shape[0] == 3
    assert output.shape[1] == 4
    assert output.numpy()[0, 0] == 2.0
    assert output.numpy()[1, 0] == 4.0
    assert output.numpy()[2, 0] == 6.0
    assert output.numpy()[0, 1] == 1.0
    assert output.numpy()[1, 1] == 3.0
    assert output.numpy()[2, 1] == 5.0
    assert output.numpy()[0, 2] == 2.0
    assert output.numpy()[1, 2] == 4.0
    assert output.numpy()[2, 2] == 6.0
    assert output.numpy()[0, 3] == 1.0
    assert output.numpy()[1, 3] == 3.0
    assert output.numpy()[2, 3] == 5.0


def test_cat():
    """Tests cat"""
    pf.set_backend("pytorch")
    a = torch.randn([2, 3, 5])
    b = torch.randn([2, 3, 5])
    val = ops.cat([a, b])
    assert val.ndim == 3
    assert val.shape[0] == 4
    assert val.shape[1] == 3
    assert val.shape[2] == 5
    val = ops.cat([a, b], axis=1)
    assert val.ndim == 3
    assert val.shape[0] == 2
    assert val.shape[1] == 6
    assert val.shape[2] == 5
    val = ops.cat([a, b], axis=2)
    assert val.ndim == 3
    assert val.shape[0] == 2
    assert val.shape[1] == 3
    assert val.shape[2] == 10


def test_additive_logistic_transform():
    """Tests additive_logistic_transform"""
    pf.set_backend("pytorch")
    a = torch.randn([2, 3, 5])
    val = ops.additive_logistic_transform(a)
    assert val.ndim == 3
    assert val.shape[0] == 2
    assert val.shape[1] == 3
    assert val.shape[2] == 6


def test_insert_col_of():
    """Tests add_col_of"""
    pf.set_backend("pytorch")
    a = torch.randn([2, 3, 5])
    val = ops.insert_col_of(a, 1)
    assert val.ndim == 3
    assert val.shape[0] == 2
    assert val.shape[1] == 3
    assert val.shape[2] == 6
    assert (val[:, :, 0] == 1).all()
