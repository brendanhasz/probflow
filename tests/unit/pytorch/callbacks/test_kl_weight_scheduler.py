import matplotlib.pyplot as plt
import pytest
from get_model_and_data import get_model_and_data

from probflow.callbacks import KLWeightScheduler


def test_KLWeightScheduler(plot):

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # should error w/ invalid args
    with pytest.raises(TypeError):
        kls = KLWeightScheduler("lala")
    with pytest.raises(TypeError):
        kls = KLWeightScheduler(lambda x: "lala")

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
