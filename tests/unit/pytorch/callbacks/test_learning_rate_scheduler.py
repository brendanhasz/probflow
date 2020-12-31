import matplotlib.pyplot as plt
import pytest
from get_model_and_data import get_model_and_data

from probflow.callbacks import LearningRateScheduler


def test_LearningRateScheduler(plot):

    # Get a model and data
    my_model, x, y = get_model_and_data()

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
    for g in my_model._optimizer.param_groups:
        assert g["lr"] == 1e-3  # check optmizer is actually updating
    lrs.plot()
    if plot:
        plt.show()

    # should error w/ invalid args
    with pytest.raises(TypeError):
        lrs = LearningRateScheduler("lala")
    with pytest.raises(TypeError):
        lrs = LearningRateScheduler(lambda x: "lala")
