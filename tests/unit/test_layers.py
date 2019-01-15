
# TODO: import as module and use absolute imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np

from probflow import Add, Sub, Mul, Div, Abs, Exp, Log


def test_add_layer():
    """Tests probflow.layers.Add"""

    # Float/int inputs
    l1 = Add(3.0, 4)
    l1.build()
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 7.0

    # Numpy array inputs
    a = np.array([[1], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Add(a, b)
    l2.build()
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0] == 4.0
    assert l2.built_obj[1] == 6.0

    # With another Layer as input
    l3 = Add(Add(3.0, 4), Add(a, b))
    l3.build()
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0] == 11.0
    assert l3.built_obj[1] == 13.0

    # With a tf.Tensor or tf.Variable as input
    # TODO

    # With a Variable as input
    # TODO


def test_sub_layer():
    """Tests probflow.layers.Sub"""

    # Float/int inputs
    l1 = Sub(4.0, 3)
    l1.build()
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 1.0

    # Numpy array inputs
    a = np.array([[0], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Sub(b, a)
    l2.build()
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0] == 3.0
    assert l2.built_obj[1] == 2.0

    # With another Layer as input
    l3 = Sub(Add(3.0, 4), Add(a, b))
    l3.build()
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0] == -4.0
    assert l3.built_obj[1] == -1.0

    # With a tf.Tensor or tf.Variable as input
    # TODO

    # With a Variable as input
    # TODO


def test_mul_layer():
    """Tests probflow.layers.Mul"""

    # Float/int inputs
    l1 = Mul(3.0, 4)
    l1.build()
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 12.0

    # Numpy array inputs
    a = np.array([[1], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Mul(a, b)
    l2.build()
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0] == 3.0
    assert l2.built_obj[1] == 8.0

    # With another Layer as input
    l3 = Mul(Add(3.0, 4), Add(a, b))
    l3.build()
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0] == 28.0
    assert l3.built_obj[1] == 42.0

    # With a tf.Tensor or tf.Variable as input
    # TODO

    # With a Variable as input
    # TODO


def test_div_layer():
    """Tests probflow.layers.Add"""

    # Float/int inputs
    l1 = Div(3.0, 4)
    l1.build()
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 0.75

    # Numpy array inputs
    a = np.array([[1], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Div(a, b)
    l2.build()
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0] == 1.0/3.0
    assert l2.built_obj[1] == 0.5

    # With another Layer as input
    l3 = Div(Add(3.0, 4), Add(a, b))
    l3.build()
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0] == 7.0/4
    assert l3.built_obj[1] == 7.0/6

    # With a tf.Tensor or tf.Variable as input
    # TODO

    # With a Variable as input
    # TODO


# TODO: Abs


# TODO: Exp


# TODO: Log


def test_layer_ops_overloading():
    """Tests that the basic arithmetic ops (__add__, etc) are overloaded"""

    # Two layers to work with
    l1 = Add(1.0, 2.0)
    l2 = Add(1.0, 1.0)

    # Add
    l3 = l1 + l2
    l3.build()
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 5.0

    # Sub
    l3 = l1 - l2
    l3.build()
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 1.0

    # Mult
    l3 = l1 * l2
    l3.build()
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 6.0

    # Div
    l3 = l1 / l2
    l3.build()
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 3.0/2.0


# TODO: don't need this w/ py.test
if __name__ == '__main__':
    test_add_layer()
    #test_sub_layer()
    test_layer_ops_overloading()