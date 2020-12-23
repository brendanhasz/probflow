from get_model_and_data import get_model_and_data

from probflow.callbacks import Callback


def test_Callback(plot):

    my_model, x, y = get_model_and_data()

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
