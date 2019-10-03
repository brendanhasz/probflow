"""Tests the probflow.models module when backend = tensorflow"""



import pytest

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.core.settings import Sampling
import probflow.core.ops as O
from probflow.distributions import Normal
from probflow.distributions import Poisson
from probflow.distributions import Bernoulli
from probflow.parameters import *
from probflow.modules import *
from probflow.models import Model
from probflow.models import ContinuousModel, DiscreteModel, CategoricalModel
from probflow.data import DataGenerator
from probflow.data import make_generator



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_Model_0D():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter(name='Weight')
            self.bias = Parameter(name='Bias')
            self.std = ScaleParameter(name='Std')

        def __call__(self, x):
            return Normal(x*self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Fit the model
    x = np.random.randn(100).astype('float32')
    y = -x + 1
    my_model.fit(x, y, batch_size=5, epochs=3)

    # Shouldn't be training
    assert my_model._is_training is False

    # Should be able to set learning rate
    lr = my_model._learning_rate
    my_model.set_learning_rate(lr+1.0)
    assert lr != my_model._learning_rate

    # but error w/ wrong type
    with pytest.raises(TypeError):
        my_model.set_learning_rate('asdf')

    # Should be able to set learning rate
    assert my_model._kl_weight == 1.0
    my_model.set_kl_weight(2.0)
    assert my_model._kl_weight == 2.0

    # but error w/ wrong type
    with pytest.raises(TypeError):
        my_model.set_kl_weight('asdf')

    # predictive samples
    samples = my_model.predictive_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    
    # aleatoric samples
    samples = my_model.aleatoric_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # epistemic samples
    samples = my_model.epistemic_sample(x[:30], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30

    # predict
    samples = my_model.predict(x[:30])
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 30

    # predict using the mode instead of the mean (same for normal dists)
    samples = my_model.predict(x[:30], method='mode')
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 30

    # metric
    metric = my_model.metric('mae', x[:30], y[:30])
    assert isinstance(metric, np.floating)
    metric = my_model.metric('mse', x[:30], y[:30])
    assert isinstance(metric, np.floating)
    assert metric >= 0
    
    # posterior_mean w/ no args should return all params
    val = my_model.posterior_mean()
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    
    # posterior_mean w/ str should return value of that param
    val = my_model.posterior_mean('Weight')
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    
    # posterior_mean w/ list of params should return only those params
    val = my_model.posterior_mean(['Weight', 'Std'])
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)

    # posterior_sample w/ no args should return all params
    val = my_model.posterior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    assert all(val[v].shape[1] == 1 for v in val)
    
    # posterior_sample w/ str should return sample of that param
    val = my_model.posterior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 2
    assert val.shape[0] == 20
    assert val.shape[1] == 1
    
    # posterior_sample w/ list of params should return only those params
    val = my_model.posterior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    assert all(val[v].shape[1] == 1 for v in val)

    # posterior_ci should return confidence intervals of all params by def
    val = my_model.posterior_ci(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 1 for v in val)
    assert all(val[v][1].ndim == 1 for v in val)
    assert all(val[v][0].shape[0] == 1 for v in val)
    assert all(val[v][1].shape[0] == 1 for v in val)

    # posterior_ci should return ci of only 1 if passed str
    val = my_model.posterior_ci('Weight', n=20)
    assert isinstance(val, tuple)
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)

    # posterior_ci should return specified cis if passed list of params
    val = my_model.posterior_ci(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 1 for v in val)
    assert all(val[v][1].ndim == 1 for v in val)
    assert all(val[v][0].shape[0] == 1 for v in val)
    assert all(val[v][1].shape[0] == 1 for v in val)

    # prior_sample w/ no args should return all params
    val = my_model.prior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    
    # prior_sample w/ str should return sample of that param
    val = my_model.prior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    assert val.shape[0] == 20
    
    # prior_sample w/ list of params should return only those params
    val = my_model.prior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)

    # log_prob should return log prob of each sample by default
    probs = my_model.log_prob(x[:30], y[:30])
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 30

    # log_prob should return sum if individually = False
    s_prob = my_model.log_prob(x[:30], y[:30], individually=False)
    assert isinstance(s_prob, np.floating)
    assert s_prob == np.sum(probs)

    # log_prob should return samples w/ distribution = True
    probs = my_model.log_prob(x[:30], y[:30], n=10, distribution=True)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 30
    assert probs.shape[1] == 10

    # log_prob should return samples w/ distribution = True
    probs = my_model.log_prob(x[:30], y[:30], n=10,
                               distribution=True, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 10

    # prob should return prob of each sample by default
    probs = my_model.prob(x[:30], y[:30])
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 30
    assert np.all(probs >= 0)

    # prob should return sum if individually = False
    s_prob = my_model.prob(x[:30], y[:30], individually=False)
    assert isinstance(s_prob, np.floating)

    # prob should return samples w/ distribution = True
    probs = my_model.prob(x[:30], y[:30], n=10, distribution=True)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 30
    assert probs.shape[1] == 10
    assert np.all(probs >= 0)

    # prob should return samples w/ distribution = True
    probs = my_model.prob(x[:30], y[:30], n=10,
                               distribution=True, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 10
    assert np.all(probs >= 0)

    # Save the model
    my_model.save('test_model.dat')



def test_Model_DataGenerators():
    """Tests the probflow.models.Model sampling/predictive methods when
    passed DataGenerators"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter(name='Weight')
            self.bias = Parameter(name='Bias')
            self.std = ScaleParameter(name='Std')

        def __call__(self, x):
            return Normal(x*self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Make a DataGenerator
    x = np.random.randn(100).astype('float32')
    y = -x + 1
    data = DataGenerator(x, y, batch_size=5)

    # Fit the model
    my_model.fit(data, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100
    
    # aleatoric samples
    samples = my_model.aleatoric_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # epistemic samples
    samples = my_model.epistemic_sample(data, n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 100

    # predict
    samples = my_model.predict(data)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.shape[0] == 100

    # metric
    metric = my_model.metric('mae', data)
    assert isinstance(metric, np.floating)
    metric = my_model.metric('mse', data)
    assert isinstance(metric, np.floating)
    assert metric >= 0
    


def test_Model_1D():
    """Tests the probflow.models.Model abstract base class"""

    class MyModel(Model):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')
            self.std = ScaleParameter([1, 1], name='Std')

        def __call__(self, x):
            return Normal(x@self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1
    
    # aleatoric samples
    samples = my_model.aleatoric_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # epistemic samples
    samples = my_model.epistemic_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # predict
    samples = my_model.predict(x[:30, :])
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 30
    assert samples.shape[1] == 1

    # metric
    metric = my_model.metric('mse', x[:30, :], y[:30, :])
    assert isinstance(metric, np.floating)
    metric = my_model.metric('mae', x[:30, :], y[:30, :])
    assert isinstance(metric, np.floating)
    assert metric >= 0
    
    # posterior_mean w/ no args should return all params
    val = my_model.posterior_mean()
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert val['Weight'].shape[0] == 5
    assert val['Weight'].shape[1] == 1
    assert val['Bias'].shape[0] == 1
    assert val['Bias'].shape[1] == 1
    assert val['Std'].shape[0] == 1
    assert val['Std'].shape[1] == 1
    
    # posterior_mean w/ str should return value of that param
    val = my_model.posterior_mean('Weight')
    assert isinstance(val, np.ndarray)
    assert val.ndim == 2
    assert val.shape[0] == 5
    assert val.shape[1] == 1
    
    # posterior_mean w/ list of params should return only those params
    val = my_model.posterior_mean(['Weight', 'Std'])
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 2 for v in val)
    assert val['Weight'].shape[0] == 5
    assert val['Weight'].shape[1] == 1
    assert val['Std'].shape[0] == 1
    assert val['Std'].shape[1] == 1

    # posterior_sample w/ no args should return all params
    val = my_model.posterior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 3 for v in val)
    assert val['Weight'].shape[0] == 20
    assert val['Weight'].shape[1] == 5
    assert val['Weight'].shape[2] == 1
    assert val['Bias'].shape[0] == 20
    assert val['Bias'].shape[1] == 1
    assert val['Bias'].shape[2] == 1
    assert val['Std'].shape[0] == 20
    assert val['Std'].shape[1] == 1
    assert val['Std'].shape[2] == 1
    
    # posterior_sample w/ str should return sample of that param
    val = my_model.posterior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 3
    assert val.shape[0] == 20
    assert val.shape[1] == 5
    assert val.shape[2] == 1
    
    # posterior_sample w/ list of params should return only those params
    val = my_model.posterior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 3 for v in val)
    assert val['Weight'].shape[0] == 20
    assert val['Weight'].shape[1] == 5
    assert val['Weight'].shape[2] == 1
    assert val['Std'].shape[0] == 20
    assert val['Std'].shape[1] == 1
    assert val['Std'].shape[2] == 1

    # posterior_ci should return confidence intervals of all params by def
    val = my_model.posterior_ci(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 2 for v in val)
    assert all(val[v][1].ndim == 2 for v in val)
    for i in range(1):
        assert val['Weight'][i].shape[0] == 5
        assert val['Weight'][i].shape[1] == 1
        assert val['Bias'][i].shape[0] == 1
        assert val['Bias'][i].shape[1] == 1
        assert val['Std'][i].shape[0] == 1
        assert val['Std'][i].shape[1] == 1

    # posterior_ci should return ci of only 1 if passed str
    val = my_model.posterior_ci('Weight', n=20)
    assert isinstance(val, tuple)
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)

    # posterior_ci should return specified cis if passed list of params
    val = my_model.posterior_ci(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], tuple) for v in val)
    assert all(isinstance(val[v][0], np.ndarray) for v in val)
    assert all(isinstance(val[v][1], np.ndarray) for v in val)
    assert all(val[v][0].ndim == 2 for v in val)
    assert all(val[v][1].ndim == 2 for v in val)
    for i in range(1):
        assert val['Weight'][i].shape[0] == 5
        assert val['Weight'][i].shape[1] == 1
        assert val['Std'][i].shape[0] == 1
        assert val['Std'][i].shape[1] == 1
    
    # prior_sample w/ no args should return all params
    val = my_model.prior_sample(n=20)
    assert isinstance(val, dict)
    assert len(val) == 3
    assert 'Weight' in val
    assert 'Bias' in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)
    
    # prior_sample w/ str should return sample of that param
    val = my_model.prior_sample('Weight', n=20)
    assert isinstance(val, np.ndarray)
    assert val.ndim == 1
    assert val.shape[0] == 20
    
    # prior_sample w/ list of params should return only those params
    val = my_model.prior_sample(['Weight', 'Std'], n=20)
    assert isinstance(val, dict)
    assert len(val) == 2
    assert 'Weight' in val
    assert 'Bias' not in val
    assert 'Std' in val
    assert all(isinstance(val[v], np.ndarray) for v in val)
    assert all(val[v].ndim == 1 for v in val)
    assert all(val[v].shape[0] == 20 for v in val)



def test_generative_Model():
    """Tests the probflow.models.Model w/ a generative model (only x)"""

    class MyModel(Model):

        def __init__(self):
            self.mean = Parameter([1], name='Mean')
            self.std = ScaleParameter([1], name='Std')

        def __call__(self):
            return Normal(self.mean(), self.std())

    # Instantiate the model
    model = MyModel()

    # Data
    X = np.random.randn(100, 1)

    # Fit the model
    model.fit(X, batch_size=10, epochs=3)

    # predictive samples
    samples = model.predictive_sample(n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 2
    assert samples.shape[0] == 50
    assert samples.shape[1] == 1

    # log_prob
    y = np.random.randn(10, 1)
    probs = model.log_prob(y)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 10
    assert probs.shape[1] == 1

    probs = model.log_prob(y, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert probs.shape[0] == 1

    probs = model.log_prob(y, distribution=True, n=11)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 3
    assert probs.shape[0] == 10
    assert probs.shape[1] == 1
    assert probs.shape[2] == 11

    probs = model.log_prob(y, distribution=True, n=11, individually=False)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[0] == 1
    assert probs.shape[1] == 11



def test_Model_nesting():
    """Tests Model when it contains Modules and sub-modules"""

    class MyModule(Module):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')

        def __call__(self, x):
            return x@self.weight() + self.bias()

    class MyModel(Model):

        def __init__(self):
            self.module = MyModule()
            self.std = ScaleParameter([1, 1], name='Std')

        def __call__(self, x):
            return Normal(self.module(x), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Shouldn't be training
    assert my_model._is_training is False

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=3)

    # predictive samples
    samples = my_model.predictive_sample(x[:30, :], n=50)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 3
    assert samples.shape[0] == 50
    assert samples.shape[1] == 30
    assert samples.shape[2] == 1

    # kl loss should be greater for outer model
    assert my_model.kl_loss().numpy() > my_model.module.kl_loss().numpy()



def test_ContinuousModel(plot):
    """Tests probflow.models.ContinuousModel"""

    class MyModel(ContinuousModel):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')
            self.std = ScaleParameter([1, 1], name='Std')

        def __call__(self, x):
            return Normal(x@self.weight() + self.bias(), self.std())

    # Instantiate the model
    model = MyModel()

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    
    # Fit the model
    model.fit(x, y, batch_size=50, epochs=100, lr=0.01)

    # predictive intervals
    lb, ub = model.predictive_interval(x[:22, :])
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 22
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 22
    assert ub.shape[1] == 1

    # predictive intervals lower ci
    llb = model.predictive_interval(x[:22, :], side='lower')
    assert isinstance(llb, np.ndarray)
    assert llb.ndim == 2
    assert llb.shape[0] == 22
    assert llb.shape[1] == 1
    assert np.all(llb<=ub)

    # predictive intervals upper ci
    uub = model.predictive_interval(x[:22, :], side='upper')
    assert isinstance(uub, np.ndarray)
    assert uub.ndim == 2
    assert uub.shape[0] == 22
    assert uub.shape[1] == 1
    assert np.all(uub>=lb)
    assert np.all(uub>=llb)

    # aleatoric intervals
    lb, ub = model.aleatoric_interval(x[:23, :])
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 23
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 23
    assert ub.shape[1] == 1

    # epistemic intervals
    lb, ub = model.epistemic_interval(x[:24, :])
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 24
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 24
    assert ub.shape[1] == 1

    # posterior predictive plot with one sample
    model.pred_dist_plot(x[:1, :])
    if plot:
        plt.title('Should be one dist on one subfig')
        plt.show()

    # posterior predictive plot with one sample, showing ci
    model.pred_dist_plot(x[:1, :], ci=0.95, style='hist')
    if plot:
        plt.title('Should be one dist on one subfig, w/ ci=0.95')
        plt.show()

    # posterior predictive plot with two samples
    model.pred_dist_plot(x[:2, :])
    if plot:
        plt.title('Should be two dists on one subfig')
        plt.show()

    # posterior predictive plot with two samples, two subfigs
    model.pred_dist_plot(x[:2, :], individually=True)
    if plot:
        plt.title('Should be two dists on two subfigs')
        plt.show()

    # posterior predictive plot with six samples, 6 subfigs, 2 cols
    model.pred_dist_plot(x[:6, :], individually=True, cols=2)
    if plot:
        plt.title('Should be 6 dists, 6 subfigs, 2 cols')
        plt.show()

    # predictive prc
    prcs = model.predictive_prc(x[:7, :], y[:7, :])
    assert isinstance(prcs, np.ndarray)
    assert prcs.ndim == 2
    assert prcs.shape[0] == 7
    assert prcs.shape[1] == 1

    with pytest.raises(TypeError):
        prcs = model.predictive_prc(x[:7, :], None)

    # predictive distribution covered for each sample
    cov = model.pred_dist_covered(x[:11, :], y[:11, :])
    assert isinstance(cov, np.ndarray)
    assert cov.ndim == 2
    assert cov.shape[0] == 11
    assert cov.shape[1] == 1
    
    with pytest.raises(ValueError):
        cov = model.pred_dist_covered(x, y, n=-1)
    with pytest.raises(ValueError):
        cov = model.pred_dist_covered(x, y, ci=-0.1)
    with pytest.raises(ValueError):
        cov = model.pred_dist_covered(x, y, ci=1.1)

    # predictive distribution covered for each sample
    cov = model.pred_dist_coverage(x[:11, :], y[:11, :])
    assert isinstance(cov, np.float)

    # plot coverage by
    xo, co = model.coverage_by(x[:, :1], x, y)
    assert isinstance(xo, np.ndarray)
    assert isinstance(co, np.ndarray)
    if plot:
        plt.title('should be coverage by plot')
        plt.show()

    # r squared
    r2 = model.r_squared(x, y, n=21)
    assert isinstance(r2, np.ndarray)
    assert r2.ndim == 2
    assert r2.shape[0] == 21
    assert r2.shape[1] == 1

    # r squared with a DataGenerator
    dg = make_generator(x, y)
    r2 = model.r_squared(dg, n=22)
    assert isinstance(r2, np.ndarray)
    assert r2.ndim == 2
    assert r2.shape[0] == 22
    assert r2.shape[1] == 1

    # plot the r2 dist
    model.r_squared_plot(x, y, style='hist')
    if plot:
        plt.title('should be r2 dist')
        plt.show()

    # residuals
    res = model.residuals(x, y)
    assert isinstance(res, np.ndarray)
    assert res.ndim == 2
    assert res.shape[0] == 100
    assert res.shape[1] == 1

    # plot the distribution of residuals
    model.residuals_plot(x, y)
    if plot:
        plt.title('should be residuals dist')
        plt.show()



def test_DiscreteModel(plot):
    """Tests probflow.models.DiscreteModel"""

    class MyModel(DiscreteModel):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')

        def __call__(self, x):
            return Poisson(tf.nn.softplus(x@self.weight() + self.bias()))

    # Instantiate the model
    model = MyModel()

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = np.round(np.exp(x@w + 1))
    
    # Fit the model
    model.fit(x, y, batch_size=50, epochs=100, lr=0.1)

    # plot the predictive dist
    model.pred_dist_plot(x[:1, :])
    if plot:
        plt.title('should be one discrete dist')
        plt.show()

    model.pred_dist_plot(x[:3, :])
    if plot:
        plt.title('should be three discrete dists')
        plt.show()

    model.pred_dist_plot(x[:3, :], cols=2)
    if plot:
        plt.title('should be three discrete dists, two cols')
        plt.show()

    # r_squared shouldn't work!
    with pytest.raises(RuntimeError):
        model.r_squared(x)

    # r_squared shouldn't work!
    with pytest.raises(RuntimeError):
        model.r_squared_plot(x)

    class MyModel(DiscreteModel):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')

        def __call__(self, x):
            return Normal(x, 1.0)

    # Instantiate the model
    model = MyModel()

    # Shouldn't work with non-discrete/scalar outputs
    with pytest.raises(NotImplementedError):
        model.pred_dist_plot(x[:1, :])


def test_CategoricalModel(plot):
    """Tests probflow.models.CategoricalModel"""

    class MyModel(CategoricalModel):

        def __init__(self):
            self.weight = Parameter([5, 1], name='Weight')
            self.bias = Parameter([1, 1], name='Bias')

        def __call__(self, x):
            return Bernoulli(x@self.weight() + self.bias())

    # Instantiate the model
    model = MyModel()

    # Data
    x = np.random.randn(100, 5).astype('float32')
    w = np.random.randn(5, 1).astype('float32')
    y = x@w + 1
    y = 1.0/(1.+np.exp(-y))
    y = np.round(y)
    
    # Fit the model
    model.fit(x, y, batch_size=50, epochs=100, lr=0.1)

    # plot the predictive dist
    model.pred_dist_plot(x[:1, :])
    if plot:
        plt.title('should be one binary dist')
        plt.show()