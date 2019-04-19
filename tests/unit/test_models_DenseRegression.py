"""Tests probflow.models.DenseRegression model"""


import numpy as np
import tensorflow as tf

from probflow.layers import Input
from probflow.models import DenseRegression



N = 8
D = 4
EPOCHS = 10



def test_models_DenseRegression():
    """Tests probflow.models.DenseRegression"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.sum(x*w, axis=1) + b + noise

    # Model 
    model = DenseRegression()

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



def test_models_DenseRegression_multiple():
    """Tests probflow.models.DenseRegression w/ multiple models"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.sum(x*w, axis=1) + b + noise

    # Models
    model1 = DenseRegression()
    model2 = DenseRegression()

    # Fit the 1st model
    model1.fit(x, y, epochs=EPOCHS)

    # Fit the 2nd model
    model2.fit(x, y, epochs=EPOCHS)



def test_models_DenseRegression_multilayer():
    """Tests probflow.models.DenseRegression w/ multiple layers"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.sum(x*w, axis=1) + b + noise

    # Model 
    model = DenseRegression(units=[5, 1])

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



def test_models_DenseRegression_data():
    """Tests probflow.models.DenseRegression data arg"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.sum(x*w, axis=1) + b + noise

    # Model 
    data = Input()
    model = DenseRegression(data=data)

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



def test_models_DenseRegression_activation():
    """Tests probflow.models.DenseRegression activation arg"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.sum(x*w, axis=1) + b + noise

    # Model 
    model = DenseRegression(activation=tf.nn.relu6)

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



# TODO: test batch_norm arg works correctly
