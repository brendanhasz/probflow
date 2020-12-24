import numpy as np
from get_model_and_data import get_model_and_data

from probflow.callbacks import MonitorParameter


def test_MonitorParameter(plot):

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test MonitorParameter
    mp = MonitorParameter("Weight")
    my_model.fit(x, y, batch_size=5, epochs=10, callbacks=[mp])
    assert isinstance(mp.current_epoch, int)
    assert mp.current_epoch == 10
    assert isinstance(mp.current_params, np.ndarray)
    assert mp.current_params
    assert isinstance(mp.epochs, list)
    assert len(mp.epochs) == 10
    assert isinstance(mp.parameter_values, list)
    assert len(mp.parameter_values) == 10
