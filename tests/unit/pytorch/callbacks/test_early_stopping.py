import numpy as np
import pytest
from get_model_and_data import get_model_and_data

from probflow.callbacks import (
    EarlyStopping,
    MonitorELBO,
    MonitorMetric,
    MonitorParameter,
)


def test_EarlyStopping():

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test EarlyStopping
    es = EarlyStopping(lambda: 3, patience=5)
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[es])
    assert isinstance(es.count, int)
    assert es.count == 6

    # should error w/ invalid args
    with pytest.raises(TypeError):
        es = EarlyStopping(lambda: 3, patience=1.1)
    with pytest.raises(ValueError):
        es = EarlyStopping(lambda: 3, patience=-1)
    with pytest.raises(TypeError):
        es = EarlyStopping("lala")


def test_multiple_callbacks():

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test multiple callbacks at the same time
    mp = MonitorParameter("Weight")
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


def test_EarlyStopping_given_MonitorMetric():

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test EarlyStopping
    mm = MonitorMetric("mae", x[:5], y[:5])
    es = EarlyStopping(mm, patience=5)
    my_model.fit(x, y, batch_size=5, epochs=3, callbacks=[mm, es])


def test_EarlyStopping_given_MonitorELBO():

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test EarlyStopping
    me = MonitorELBO()
    es = EarlyStopping(me, patience=5)
    my_model.fit(x, y, batch_size=5, epochs=3, callbacks=[me, es])
