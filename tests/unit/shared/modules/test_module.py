import numpy as np

import probflow.utils.ops as O
from probflow.distributions.normal import Normal
from probflow.modules import Module
from probflow.parameters import Parameter
from probflow.utils.casting import to_numpy, to_tensor, to_default_dtype
from probflow.utils.settings import Sampling
from probflow.utils.validation import is_backend_tensor


def test_Module():
    """Tests the Module abstract base class."""

    class TestModule(Module):
        def __init__(self):
            self.p1 = Parameter(name="TestParam1")
            self.p2 = Parameter(name="TestParam2", shape=[5, 4])

        def __call__(self, x):
            x = to_tensor(x)
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
    assert all(is_backend_tensor(v) for v in var_list)

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert is_backend_tensor(kl_loss)
    assert kl_loss.ndim == 0

    # calling a module should return a tensor
    x = O.randn([5])
    sample1 = the_module(x)
    assert is_backend_tensor(sample1)
    assert sample1.ndim == 1
    assert sample1.shape[0] == 5

    # should be the same when sampling is off
    sample2 = the_module(x)
    assert np.all(to_numpy(sample1) == to_numpy(sample2))

    # outputs should be different when sampling is on
    with Sampling(n=1):
        sample1 = the_module(x)
        sample2 = the_module(x)
    assert np.all(to_numpy(sample1) != to_numpy(sample2))

    # bayesian_update should update all params in the module
    assert np.all(
        to_numpy(the_module.p1.prior.loc)
        != to_numpy(the_module.p1.posterior.loc)
    )
    assert np.all(
        to_numpy(the_module.p2.prior.scale)
        != to_numpy(the_module.p2.posterior.scale)
    )
    the_module.bayesian_update()
    assert np.all(
        to_numpy(the_module.p1.prior.loc)
        == to_numpy(the_module.p1.posterior.loc)
    )
    assert np.all(
        to_numpy(the_module.p2.prior.scale)
        == to_numpy(the_module.p2.posterior.scale)
    )


def test_Module_nesting():
    """Tests creating Modules within Modules."""

    # Non-nested module
    class TestModule(Module):
        def __init__(self):
            self.p1 = Parameter(name="TestParam1")
            self.p2 = Parameter(name="TestParam2", shape=[5, 4])

        def __call__(self, x):
            return O.sum(self.p2(), axis=None) + x * self.p1()

    # A module which contains sub-modules
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
    assert all(is_backend_tensor(v) for v in var_list)

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert is_backend_tensor(kl_loss)
    assert kl_loss.ndim == 0

    # parent module's loss should be greater than child module's
    assert to_numpy(the_module.kl_loss()) > to_numpy(the_module.mod.kl_loss())

    # calling a module should return a tensor
    x = O.randn([5])
    sample1 = the_module(x)
    assert is_backend_tensor(sample1)
    assert sample1.ndim == 1
    assert sample1.shape[0] == 5

    # of the appropriate size
    x = O.randn([5, 4])
    sample1 = the_module(x)
    assert is_backend_tensor(sample1)
    assert sample1.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 4


def test_Module_lists_and_dicts():
    """Tests creating Modules which have list/dict attribs w/ params."""

    # Module which contains lists/dicts w/ parameters
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
                O.ones([x.shape[0], 1])
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
    the_module.add_kl_loss(to_default_dtype(3.145))
    assert np.isclose(to_numpy(the_module.kl_loss_batch()), 3.145)

    # And should also be able to add kl losses from two distributions
    the_module.reset_kl_loss()
    d1 = Normal(0.0, 1.0)
    d2 = Normal(1.0, 1.0)
    assert the_module.kl_loss_batch() == 0
    the_module.add_kl_loss_between(d1, d2)
    assert to_numpy(the_module.kl_loss_batch()) > 0.0


def test_Module_lists_and_dicts_nesting():
    """Tests creating Modules which have list/dict attribs w/ params+modules."""

    # A basic Module
    class TestModule1(Module):
        def __init__(self):
            self.p1 = Parameter(name="TestParam1")
            self.p2 = Parameter(name="TestParam2", shape=[5, 4])

        def __call__(self, x):
            return O.sum(self.p2(), axis=None) + x * self.p1()

    # A second basic module
    class TestModule2(Module):
        def __init__(self):
            self.p1 = Parameter(name="TestParam3")
            self.p2 = Parameter(name="TestParam4", shape=[5, 4])

        def __call__(self, x):
            return O.sum(self.p2(), axis=None) + x * self.p1()

    # Module which contains lists/dicts w/ parameters
    class TestModule3(Module):
        def __init__(self):
            self.a_list = [
                Parameter(name="TestParam5"),
                Parameter(name="TestParam6"),
                TestModule1(),
            ]
            self.a_dict = {
                "b": Parameter(name="TestParam7"),
                "a": Parameter(name="TestParam8"),
                "c": TestModule2(),
            }

        def __call__(self, x):
            return (
                O.ones([x.shape[0], 1])
                + self.a_list[0]()
                + self.a_list[1]()
                + self.a_list[2]()
                + self.a_dict["a"]()
                + self.a_dict["b"]()
                + self.a_dict["c"]()
            )

    the_module = TestModule3()

    # parameters should return a list of all the parameters
    param_list = the_module.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 8
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in param_list]
    assert "TestParam1" in param_names
    assert "TestParam2" in param_names
    assert "TestParam3" in param_names
    assert "TestParam4" in param_names
    assert "TestParam5" in param_names
    assert "TestParam6" in param_names
    assert "TestParam7" in param_names
    assert "TestParam8" in param_names

    # trainable_variables should return list of all variables in the model
    var_list = the_module.trainable_variables
    assert isinstance(var_list, list)
    assert len(var_list) == 16
    assert all(is_backend_tensor(v) for v in var_list)

    # n_params property should include all params in submodules
    nparams = the_module.n_parameters
    assert isinstance(nparams, int)
    assert nparams == 46

    # n_variables property should include all variables in submodules
    nvars = the_module.n_variables
    assert isinstance(nvars, int)
    assert nvars == 92

    # kl_loss should return sum of all the kl losses
    kl_loss = the_module.kl_loss()
    assert is_backend_tensor(kl_loss)
    assert kl_loss.ndim == 0

    # parent module's loss should be greater than child module's
    assert to_numpy(the_module.kl_loss()) > to_numpy(
        the_module.a_list[2].kl_loss()
    )
    assert to_numpy(the_module.kl_loss()) > to_numpy(
        the_module.a_dict["c"].kl_loss()
    )

    # Loss should be the sum of all parameter losses w/i the module
    assert np.isclose(
        to_numpy(the_module.kl_loss()),
        (
            to_numpy(the_module.a_list[0].kl_loss())
            + to_numpy(the_module.a_list[1].kl_loss())
            + to_numpy(the_module.a_list[2].p1.kl_loss())
            + to_numpy(the_module.a_list[2].p2.kl_loss())
            + to_numpy(the_module.a_dict["a"].kl_loss())
            + to_numpy(the_module.a_dict["b"].kl_loss())
            + to_numpy(the_module.a_dict["c"].p1.kl_loss())
            + to_numpy(the_module.a_dict["c"].p2.kl_loss())
        ),
    )
