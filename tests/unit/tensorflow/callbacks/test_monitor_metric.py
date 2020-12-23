import matplotlib.pyplot as plt
import numpy as np
from get_model_and_data import get_model_and_data

from probflow.callbacks import MonitorMetric


def test_MonitorMetric(plot):

    # Get a model and data
    my_model, x, y = get_model_and_data()

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

    # Test plotting vs time, and passing kwargs to plt.plot
    mm.plot(x="time", label="model1")
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
