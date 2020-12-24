import matplotlib.pyplot as plt
import numpy as np
from get_model_and_data import get_model_and_data

from probflow.callbacks import MonitorELBO


def test_MonitorELBO(plot):

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test MontiorELBO
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

    # Test plotting vs time, and passing kwargs to plt.plot
    me.plot(x="time", label="model2")
    if plot:
        plt.show()
