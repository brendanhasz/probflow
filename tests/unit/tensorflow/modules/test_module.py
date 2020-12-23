import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import probflow.utils.ops as O
from probflow.modules import Module
from probflow.parameters import Parameter
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Module():
    """Tests the Module abstract base class"""

    class TestModule(Module):
        def __init__(self):
            self.p1 = Parameter(name="TestParam1")
            self.p2 = Parameter(name="TestParam2", shape=[5, 4])

        def __call__(self, x):
            return O.sum(self.p2(), axis=None) + x * self.p1()

    the_module = TestModule()

    # parameters should return a list of all the parameters
    param_list = the_module.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in param_list]
    assert "TestParam1" in param_names
    assert "TestParam2" in param_names

    # n_parameters property
    nparams = the_module.n_parameters
    assert isinstance(nparams, int)
    assert nparams == 21

    # n_variables property
    nvars = the_module.n_variables
    assert isinstance(nvars, int)
    assert nvars == 42

    # trainable_variables should return list of all variables in the model
    var_list = the_module.trainable_variables
    assert isinstance(var_list, list)
    assert len(var_list) == 4
    assert all(isinstance(v, tf.Variable) for v in var_list)

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # calling a module should return a tensor
    x = tf.random.normal([5])
    sample1 = the_module(x)
    assert isinstance(sample1, tf.Tensor)
    assert sample1.ndim == 1
    assert sample1.shape[0] == 5

    # should be the same when sampling is off
    sample2 = the_module(x)
    assert np.all(sample1.numpy() == sample2.numpy())

    # outputs should be different when sampling is on
    with Sampling():
        sample1 = the_module(x)
        sample2 = the_module(x)
    assert np.all(sample1.numpy() != sample2.numpy())

    # A second test module which contains sub-modules
    class TestModule2(Module):
        def __init__(self, shape):
            self.mod = TestModule()
            self.p3 = Parameter(name="TestParam3", shape=shape)

        def __call__(self, x):
            return self.mod(x) + O.sum(self.p3(), axis=None)

    the_module = TestModule2([3, 2])

    # parameters should return a list of all the parameters
    param_list = the_module.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 3
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in param_list]
    assert "TestParam1" in param_names
    assert "TestParam2" in param_names
    assert "TestParam3" in param_names

    # n_params property
    nparams = the_module.n_parameters
    assert isinstance(nparams, int)
    assert nparams == 27

    # trainable_variables should return list of all variables in the model
    var_list = the_module.trainable_variables
    assert isinstance(var_list, list)
    assert len(var_list) == 6
    assert all(isinstance(v, tf.Variable) for v in var_list)

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # parent module's loss should be greater than child module's
    assert the_module.kl_loss().numpy() > the_module.mod.kl_loss().numpy()

    # calling a module should return a tensor
    x = tf.random.normal([5])
    sample1 = the_module(x)
    assert isinstance(sample1, tf.Tensor)
    assert sample1.ndim == 1
    assert sample1.shape[0] == 5

    # of the appropriate size
    x = tf.random.normal([5, 4])
    sample1 = the_module(x)
    assert isinstance(sample1, tf.Tensor)
    assert sample1.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 4

    # Another test module which contains lists/dicts w/ parameters
    class TestModule3(Module):
        def __init__(self):
            self.a_list = [
                Parameter(name="TestParam4"),
                Parameter(name="TestParam5"),
            ]
            self.a_dict = {
                "a": Parameter(name="TestParam6"),
                "b": Parameter(name="TestParam7"),
            }

        def __call__(self, x):
            return (
                tf.ones([x.shape[0], 1])
                + self.a_list[0]()
                + self.a_list[1]()
                + self.a_dict["a"]()
                + self.a_dict["b"]()
            )

    the_module = TestModule3()

    # parameters should return a list of all the parameters
    param_list = the_module.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 4
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in param_list]
    assert "TestParam4" in param_names
    assert "TestParam5" in param_names
    assert "TestParam6" in param_names
    assert "TestParam7" in param_names

    # n_params property
    nparams = the_module.n_parameters
    assert isinstance(nparams, int)
    assert nparams == 4

    # Should be able to initialize and add kl losses
    the_module.reset_kl_loss()
    assert the_module.kl_loss_batch() == 0
    the_module.add_kl_loss(3.145)
    assert is_close(the_module.kl_loss_batch().numpy(), 3.145)

    # And should also be able to pass two dists to add_kl_loss
    the_module.reset_kl_loss()
    d1 = tfd.Normal(0.0, 1.0)
    d2 = tfd.Normal(1.0, 1.0)
    assert the_module.kl_loss_batch() == 0
    the_module.add_kl_loss(d1, d2)
    assert the_module.kl_loss_batch().numpy() > 0.0
