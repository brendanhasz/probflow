import time

from get_model_and_data import get_model_and_data

from probflow.callbacks import TimeOut


def test_TimeOut():

    # Get a model and data
    my_model, x, y = get_model_and_data()

    # Test TimeOut
    to = TimeOut(2)
    ts = time.time()
    my_model.fit(x, y, batch_size=5, epochs=10000, callbacks=[to])
    assert time.time() - ts < 4
