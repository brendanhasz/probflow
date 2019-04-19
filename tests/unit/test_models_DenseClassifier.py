"""Tests probflow.models.DenseClassifier model"""


import numpy as np
import tensorflow as tf

from probflow.layers import Input
from probflow.models import DenseClassifier



N = 8
D = 4
EPOCHS = 10



def test_models_DenseClassifier():
    """Tests probflow.models.DenseClassifier"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.round(1.0/(1.0 + np.exp(-(np.sum(x*w, axis=1) + b + noise))))

    # Model 
    model = DenseClassifier()

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



def test_models_DenseClassifier_multiple():
    """Tests probflow.models.DenseClassifier w/ multiple models"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.round(1.0/(1.0 + np.exp(-(np.sum(x*w, axis=1) + b + noise))))

    # Models
    model1 = DenseClassifier()
    model2 = DenseClassifier()

    # Fit the 1st model
    model1.fit(x, y, epochs=EPOCHS)

    # Fit the 2nd model
    model2.fit(x, y, epochs=EPOCHS)



def test_models_DenseClassifier_multilayer():
    """Tests probflow.models.DenseClassifier w/ multiple layers"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.round(1.0/(1.0 + np.exp(-(np.sum(x*w, axis=1) + b + noise))))

    # Model
    model = DenseClassifier(units=[5, 1])

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



def test_models_DenseClassifier_data():
    """Tests probflow.models.DenseClassifier data arg"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.round(1.0/(1.0 + np.exp(-(np.sum(x*w, axis=1) + b + noise))))

    # Model 
    data = Input()
    model = DenseClassifier(data=data)

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



def test_models_DenseClassifier_activation():
    """Tests probflow.models.DenseClassifier activation arg"""

    # Dummy data
    x = np.random.randn(N, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.round(1.0/(1.0 + np.exp(-(np.sum(x*w, axis=1) + b + noise))))

    # Model 
    model = DenseClassifier(activation=tf.nn.relu6)

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)



# TODO: test batch_norm arg works correctly
