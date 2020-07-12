"""Tests the probflow.callbacks module"""


import pytest
import time

import numpy as np
import matplotlib.pyplot as plt

from probflow.distributions import Normal
from probflow.parameters import *
from probflow.modules import *
from probflow.models import *
from probflow.callbacks import *


def test_Callback(plot):
    """Tests the probflow.callbacks.Callback"""

    class MyModel(Model):
        def __init__(self):
            self.weight = Parameter(name="Weight")
            self.bias = Parameter(name="Bias")
            self.std = ScaleParameter(name="Std")

        def __call__(self, x):
            return Normal(x * self.weight() + self.bias(), self.std())

    # Instantiate the model
    my_model = MyModel()

    # Some data to fit
    x = np.random.randn(100).astype("float32")
    y = -x + 1

    # Create a callback class
    class MyCallback(Callback):
        def __init__(self):
            self.count = 0

        def on_epoch_end(self):
            self.count += 1

    # Create a callback object
    my_callback = MyCallback()

    # Fit the model
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[my_callback])

    # Check callback was called and worked appropriately
    assert my_callback.count == 10

    # Test LearningRateScheduler
    lrs = LearningRateScheduler(lambda x: 1e-2 / x)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[lrs])
    assert isinstance(lrs.current_epoch, int)
    assert lrs.current_epoch == 10
    assert isinstance(lrs.current_lr, float)
    assert lrs.current_lr == 1e-3
    assert isinstance(lrs.epochs, list)
    assert len(lrs.epochs) == 10
    assert isinstance(lrs.learning_rate, list)
    assert len(lrs.learning_rate) == 10
    assert my_model._learning_rate == 1e-3
    lrs.plot()
    if plot:
        plt.show()

    # should error w/ invalid args
    with pytest.raises(TypeError):
        lrs = LearningRateScheduler("lala")
    with pytest.raises(TypeError):
        lrs = LearningRateScheduler(lambda x: "lala")

    # Test KLWeightScheduler
    kls = KLWeightScheduler(lambda x: x / 100.0)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[kls])
    assert isinstance(kls.current_epoch, int)
    assert kls.current_epoch == 10
    assert isinstance(kls.current_w, float)
    assert kls.current_w == 0.1
    assert isinstance(kls.epochs, list)
    assert len(kls.epochs) == 10
    assert isinstance(kls.kl_weights, list)
    assert len(kls.kl_weights) == 10
    assert my_model._kl_weight == 0.1
    kls.plot()
    if plot:
        plt.show()

    # should error w/ invalid args
    with pytest.raises(TypeError):
        lrs = KLWeightScheduler("lala")
    with pytest.raises(TypeError):
        lrs = KLWeightScheduler(lambda x: "lala")

    # Test MontiorMetric
    x_val = np.random.randn(100).astype("float32")
    y_val = -x_val + 1
    mm = MonitorMetric("mae", x_val, y_val, verbose=True)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[mm])
    assert isinstance(mm.current_epoch, int)
    assert mm.current_epoch == 10
    assert isinstance(mm.current_metric, np.floating)
    assert mm.current_metric > 0.0
    assert isinstance(mm.epochs, list)
    assert len(mm.epochs) == 10
    assert isinstance(mm.metrics, list)
    assert len(mm.metrics) == 10
    mm.plot()
    if plot:
        plt.show()

    # MontiorMetric with custom metric function
    x_val = np.random.randn(100).astype("float32")
    y_val = -x_val + 1
    fn = lambda y_true, y_pred: sum((y_true - y_pred) * (y_true - y_pred))
    mm2 = MonitorMetric(fn, x_val, y_val, verbose=True)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[mm2])
    assert isinstance(mm2.current_epoch, int)
    assert mm2.current_epoch == 10
    assert isinstance(mm2.current_metric, np.floating)
    assert mm2.current_metric > 0.0
    assert isinstance(mm2.epochs, list)
    assert len(mm2.epochs) == 10
    assert isinstance(mm2.metrics, list)
    assert len(mm2.metrics) == 10
    mm.plot()
    if plot:
        plt.show()

    # Test MontiorELBO
    x_val = np.random.randn(100).astype("float32")
    y_val = -x_val + 1
    me = MonitorELBO(verbose=True)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[me])
    assert isinstance(me.current_epoch, int)
    assert me.current_epoch == 10
    assert isinstance(me.current_elbo, np.floating)
    assert isinstance(me.epochs, list)
    assert len(me.epochs) == 10
    assert isinstance(me.elbos, list)
    assert len(me.elbos) == 10
    me.plot()
    if plot:
        plt.show()

    # Test MonitorParameter
    mp = MonitorParameter(x_val, y_val, params="Weight")
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[mp])
    assert isinstance(mp.current_epoch, int)
    assert mp.current_epoch == 10
    assert isinstance(mp.current_params, np.ndarray)
    assert mp.current_params
    assert isinstance(mp.epochs, list)
    assert len(mp.epochs) == 10
    assert isinstance(mp.parameter_values, list)
    assert len(mp.parameter_values) == 10

    # Test EarlyStopping
    es = EarlyStopping(lambda: 3, patience=5)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[es])
    assert isinstance(es.count, int)
    assert es.count == 6

    # Test TimeOut
    to = TimeOut(2)
    ts = time.time()
    my_model.fit(x, y, batch_size=5, epochs=10000, callbacks=[to])
    assert time.time() - ts < 4

    # should error w/ invalid args
    with pytest.raises(TypeError):
        es = EarlyStopping(lambda: 3, patience=1.1)
    with pytest.raises(ValueError):
        es = EarlyStopping(lambda: 3, patience=-1)
    with pytest.raises(TypeError):
        es = EarlyStopping("lala")

    # Test multiple callbacks at the same time
    mp = MonitorParameter(x_val, y_val, params="Weight")
    es = EarlyStopping(lambda: 3, patience=5)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[mp, es])
    assert isinstance(es.count, int)
    assert es.count == 6
    assert isinstance(mp.current_epoch, int)
    assert mp.current_epoch == 7
    assert isinstance(mp.current_params, np.ndarray)
    assert mp.current_params
    assert isinstance(mp.epochs, list)
    assert len(mp.epochs) == 7
    assert isinstance(mp.parameter_values, list)
