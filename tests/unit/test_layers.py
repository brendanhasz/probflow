"""Tests probflow.layers modules"""



import numpy as np
import tensorflow as tf

from probflow.layers import Add, Sub, Mul, Div, Neg, Abs, Exp, Log
from probflow.layers import Reciprocal, Sqrt, Transform
from probflow.layers import Sigmoid, Relu, Softmax
from probflow.layers import Sum, Mean, Min, Max, Prod, LogSumExp
from probflow.layers import Reshape, Cat, Dot, Matmul
from probflow.parameters import Parameter
from probflow.distributions import Normal



def isclose(a, b, tol=1e-7):
    """Returns true if a and b are w/i tol"""
    return abs(a-b) < tol



def test_add_layer():
    """Tests probflow.layers.Add"""

    # Float/int inputs
    l1 = Add(3.0, 4)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 7.0

    # Numpy array inputs
    a = np.array([[1], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Add(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0][0] == 4.0
    assert l2.built_obj[1][0] == 6.0

    # With another Layer as input
    l3 = Add(Add(3.0, 4), Add(a, b))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0] == 11.0
    assert l3.built_obj[1] == 13.0

    # With a tf.Tensor as input
    a = tf.constant([[1], [2]], dtype=tf.float32)
    b = tf.constant([[3], [4]], dtype=tf.float32)
    l2 = Add(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 4.0
    assert l2_out[1][0] == 6.0

    # With a tf.Variable as input
    a = tf.Variable([[1], [2]], dtype=tf.float32)
    b = tf.Variable([[3], [4]], dtype=tf.float32)
    l2 = Add(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 4.0
    assert l2_out[1][0] == 6.0

    # With a Parameter as input
    a = Parameter(shape=[3,4])
    b = Parameter(shape=[3,4])
    l1 = Add(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l1_out = sess.run(l1.built_obj)
    assert isinstance(l1_out, np.ndarray)
    assert l1_out.ndim == 3
    assert l1_out.shape[0] == 2
    assert l1_out.shape[1] == 3
    assert l1_out.shape[2] == 4

    # Reset the graph
    tf.reset_default_graph()


def test_sub_layer():
    """Tests probflow.layers.Sub"""

    # Float/int inputs
    l1 = Sub(4.0, 3)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 1.0

    # Numpy array inputs
    a = np.array([[0], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Sub(b, a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0] == 3.0
    assert l2.built_obj[1] == 2.0

    # With another Layer as input
    l3 = Sub(Add(3.0, 4), Add(a, b))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0][0] == 4.0
    assert l3.built_obj[1][0] == 1.0

    # With a tf.Tensor as input
    a = tf.constant([[0], [2]], dtype=tf.float32)
    b = tf.constant([[3], [4]], dtype=tf.float32)
    l2 = Sub(b, a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 3.0
    assert l2_out[1][0] == 2.0

    # With a tf.Variable as input
    a = tf.Variable([[0], [2]], dtype=tf.float32)
    b = tf.Variable([[3], [4]], dtype=tf.float32)
    l2 = Sub(b, a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 3.0
    assert l2_out[1][0] == 2.0


def test_mul_layer():
    """Tests probflow.layers.Mul"""

    # Float/int inputs
    l1 = Mul(3.0, 4)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 12.0

    # Numpy array inputs
    a = np.array([[1], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l1 = Mul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, np.ndarray)
    assert l1.built_obj.ndim == 2
    assert l1.built_obj.shape[0] == 2
    assert l1.built_obj.shape[1] == 1
    assert l1.built_obj[0] == 3.0
    assert l1.built_obj[1] == 8.0

    # With another Layer as input
    l1 = Mul(Add(3.0, 4), Add(a, b))
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, np.ndarray)
    assert l1.built_obj.ndim == 2
    assert l1.built_obj.shape[0] == 2
    assert l1.built_obj.shape[1] == 1
    assert l1.built_obj[0] == 28.0
    assert l1.built_obj[1] == 42.0

    # With a tf.Tensor as input
    a = tf.constant([[1], [2]], dtype=tf.float32)
    b = tf.constant([[3], [4]], dtype=tf.float32)
    l1 = Mul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, tf.Tensor)
    assert len(l1.built_obj.shape) == 2
    assert l1.built_obj.shape[0].value == 2
    assert l1.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l1_out = sess.run(l1.built_obj)
    assert isinstance(l1_out, np.ndarray)
    assert l1_out.ndim == 2
    assert l1_out.shape[0] == 2
    assert l1_out.shape[1] == 1
    assert l1_out[0][0] == 3.0
    assert l1_out[1][0] == 8.0

    # With a tf.Variable as input
    a = tf.Variable([[1], [2]], dtype=tf.float32)
    b = tf.Variable([[3], [4]], dtype=tf.float32)
    l2 = Mul(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 3.0
    assert l2_out[1][0] == 8.0


def test_div_layer():
    """Tests probflow.layers.Add"""

    # Float/int inputs
    l1 = Div(3.0, 4)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 0.75

    # Numpy array inputs
    a = np.array([[1], [2]]).astype('float32')
    b = np.array([[3], [4]]).astype('float32')
    l2 = Div(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0] == 1.0/3.0
    assert l2.built_obj[1] == 0.5

    # With another Layer as input
    l3 = Div(Add(3.0, 4), Add(a, b))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, np.ndarray)
    assert l3.built_obj.ndim == 2
    assert l3.built_obj.shape[0] == 2
    assert l3.built_obj.shape[1] == 1
    assert l3.built_obj[0] == 7.0/4
    assert l3.built_obj[1] == 7.0/6

    # With a tf.Tensor as input
    a = tf.constant([[1], [2]], dtype=tf.float32)
    b = tf.constant([[3], [4]], dtype=tf.float32)
    l2 = Div(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 1.0/3.0) #float32 vs 64...
    assert l2_out[1][0] == 0.5

    # With a tf.Variable as input
    a = tf.Variable([[1], [2]], dtype=tf.float32)
    b = tf.Variable([[3], [4]], dtype=tf.float32)
    l2 = Div(a, b)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 1.0/3.0) #float32 vs 64...
    assert l2_out[1][0] == 0.5


def test_neg_layer():
    """Tests probflow.layers.Neg"""

    # Int inputs
    l1 = Neg(2)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float) #should auto-convert to float
    assert l1.built_obj == -2

    # Float inputs
    l1 = Neg(3.0)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == -3.0

    # Numpy array inputs
    a = np.array([[1], [-2]]).astype('float32')
    l2 = Neg(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0][0] == -1.0
    assert l2.built_obj[1][0] == 2.0

    # With another Layer as input
    l3 = Neg(Add(3.0, 4))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == -7.0

    # With a tf.Tensor as input
    a = tf.constant([[1], [-2]], dtype=tf.float32)
    l2 = Neg(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == -1.0
    assert l2_out[1][0] == 2.0

    # With a tf.Variable as input
    a = tf.Variable([[1], [-2]], dtype=tf.float32)
    l2 = Neg(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == -1.0
    assert l2_out[1][0] == 2.0


def test_abs_layer():
    """Tests probflow.layers.Abs"""

    # Positive Int input
    l1 = Abs(2)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float) #should auto-convert to float
    assert l1.built_obj == 2

    # Negative Int input
    l1 = Abs(-2)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float) #should auto-convert to float
    assert l1.built_obj == 2

    # Positive float inputs
    l1 = Abs(3.0)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 3.0

    # Negative float inputs
    l1 = Abs(-3.0)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, float)
    assert l1.built_obj == 3.0

    # Numpy array inputs
    a = np.array([[1], [-2]]).astype('float32')
    l2 = Abs(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, np.ndarray)
    assert l2.built_obj.ndim == 2
    assert l2.built_obj.shape[0] == 2
    assert l2.built_obj.shape[1] == 1
    assert l2.built_obj[0][0] == 1.0
    assert l2.built_obj[1][0] == 2.0

    # With another (positive) Layer as input
    l3 = Abs(Sub(4.0, 3))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 1.0

    # With another (negative) Layer as input
    l3 = Abs(Sub(3.0, 5))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 2.0

    # With a tf.Tensor as input
    a = tf.constant([[1], [-2]], dtype=tf.float32)
    l2 = Abs(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 1.0
    assert l2_out[1][0] == 2.0

    # With a tf.Variable as input
    a = tf.Variable([[1], [-2]], dtype=tf.float32)
    l2 = Abs(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert l2_out[0][0] == 1.0
    assert l2_out[1][0] == 2.0


def test_exp_layer():
    """Tests probflow.layers.Exp"""

    # Int input
    l1 = Exp(1)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l1_out = sess.run(l1.built_obj)
    assert isclose(l1_out, 2.718281828459045)

    # Float inputs
    l1 = Exp(1.0)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l1_out = sess.run(l1.built_obj)
    assert isclose(l1_out, 2.718281828459045)

    # Numpy array inputs
    a = np.array([[1], [-2.0]]).astype('float32')
    l2 = Exp(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0],  2.718281828459045)
    assert isclose(l2_out[1][0], 0.1353352832366127)

    # With another Layer as input
    l3 = Exp(Add(0.3, 0.7))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l3_out = sess.run(l3.built_obj)
    assert isclose(l3_out, 2.718281828459045)

    # With a tf.Tensor as input
    a = tf.constant([[1], [-2]], dtype=tf.float32)
    l2 = Exp(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 2.718281828459045)
    assert isclose(l2_out[1][0], 0.1353352832366127)

    # With a tf.Variable as input
    a = tf.Variable([[1], [-2]], dtype=tf.float32)
    l2 = Exp(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 2.718281828459045)
    assert isclose(l2_out[1][0], 0.1353352832366127)


def test_log_layer():
    """Tests probflow.layers.Log"""

    # Int input
    l1 = Log(1)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l1_out = sess.run(l1.built_obj)
    assert l1_out == 0

    # Float inputs
    l1 = Log(2.718281828459045)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l1.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l1_out = sess.run(l1.built_obj)
    assert isclose(l1_out, 1.0)

    # Numpy array inputs
    a = np.array([[1], [2.718281828459045]]).astype('float32')
    l2 = Log(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 0.0)
    assert isclose(l2_out[1][0], 1.0)

    # With another Layer as input
    l3 = Log(Add(0.3, 0.7))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, tf.Tensor)
    with tf.Session() as sess:
        l3_out = sess.run(l3.built_obj)
    assert isclose(l3_out, 0.0)

    # With a tf.Tensor as input
    a = tf.constant([[1], [2.718281828459045]], dtype=tf.float32)
    l2 = Log(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 0.0)
    assert isclose(l2_out[1][0], 1.0)

    # With a tf.Variable as input
    a = tf.Variable([[1], [2.718281828459045]], dtype=tf.float32)
    l2 = Log(a)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l2.built_obj, tf.Tensor)
    assert len(l2.built_obj.shape) == 2
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 1
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        l2_out = sess.run(l2.built_obj)
    assert isinstance(l2_out, np.ndarray)
    assert l2_out.ndim == 2
    assert l2_out.shape[0] == 2
    assert l2_out.shape[1] == 1
    assert isclose(l2_out[0][0], 0.0)
    assert isclose(l2_out[1][0], 1.0)


def test_layer_special_methods():
    """Tests the arithmetic ops (__add__, etc) defined as special methods"""

    # Two layers to work with
    l1 = Add(1.0, 2.0)
    l2 = Add(1.0, 1.0)

    # Add
    l3 = l1 + l2
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 5.0

    # Sub
    l3 = l1 - l2
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 1.0

    # Mult
    l3 = l1 * l2
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 6.0

    # Div
    l3 = l1 / l2
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 3.0/2.0

    # Neg
    l3 = -l1
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == -3.0

    # Abs
    l3 = abs(Sub(3.0, 4))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(l3.built_obj, float)
    assert l3.built_obj == 1.0


def test_layer_broadcasting():
    """Tests broadcasting works w/ layers of different shapes"""

    # Single dimension
    a = tf.random.normal([1])
    b = tf.random.normal([2])
    l1 = Add(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert l1.built_obj.shape[0].value == 2

    # Two dimensions
    c = tf.random.normal([2, 1])
    d = tf.random.normal([2, 3])
    l2 = Add(c, d)
    l2._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert l2.built_obj.shape[0].value == 2
    assert l2.built_obj.shape[1].value == 3

    # Two layers
    l3 = Add(Add(d, c), Add(c, d))
    l3._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert l3.built_obj.shape[0].value == 2
    assert l3.built_obj.shape[1].value == 3

    # Different number of dimensions
    e = tf.random.normal([2])
    f = tf.random.normal([2, 3])
    l4 = Add(e, f)
    l4._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert l4.built_obj.shape[0].value == 2
    assert l4.built_obj.shape[1].value == 3



# TODO: 
# Reciprocal
# Sqrt
# Transform
# Sigmoid
# Relu
# Softmax
#
# Sum
# Mean
# Min
# Max
# Prod
# LogSumExp



def test_layer_reshape():
    """Tests probflow.layers.Reshape"""

    # Single dimension
    a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    l1 = Reshape(a, shape=[1, 2])
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 1
    assert l1.built_obj.shape[2].value == 2
    with tf.Session() as sess:
        out = sess.run(l1.built_obj)
    assert out[0, 0, 0] == 1.0
    assert out[0, 0, 1] == 2.0
    assert out[1, 0, 0] == 3.0
    assert out[1, 0, 1] == 4.0
    assert out[2, 0, 0] == 5.0
    assert out[2, 0, 1] == 6.0



# TODO: Cat



def test_layer_dot():
    """Tests probflow.layers.Dot"""

    # Single dimension
    a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    b = tf.constant([[7, 8], [9, 10], [11, 12]], dtype=tf.float32)
    l1 = Dot(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 2
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 1
    with tf.Session() as sess:
        out = sess.run(l1.built_obj)
    assert out[0, 0] == 23.0
    assert out[1, 0] == 67.0
    assert out[2, 0] == 127.0

    # With parameters, single dimension
    a = Parameter(shape=4)
    b = Parameter(shape=4)
    l1 = Dot(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 2
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 1

    # With parameters, multiple dimensions
    a = Parameter(shape=[4, 5])
    b = Parameter(shape=[4, 5])
    l1 = Dot(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 4
    assert l1.built_obj.shape[2].value == 1

    # With parameters, multiple dimensions, different axis
    a = Parameter(shape=[4, 5])
    b = Parameter(shape=[4, 5])
    l1 = Dot(a, b, axis=-2)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 1
    assert l1.built_obj.shape[2].value == 5

    # With parameters, multiple dimensions, different axis, no keepdims
    a = Parameter(shape=[4, 5])
    b = Parameter(shape=[4, 5])
    l1 = Dot(a, b, axis=-2, keepdims=False)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 2
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 5



def test_layer_matmul():
    """Tests probflow.layers.Matmul"""

    # Single dimension, batch size of 1
    a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    b = tf.constant([[7, 8], [9, 10]], dtype=tf.float32)
    a = tf.reshape(a, [1, 3, 2])
    b = tf.reshape(b, [1, 2, 2])
    l1 = Matmul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 1
    assert l1.built_obj.shape[1].value == 3
    assert l1.built_obj.shape[2].value == 2
    with tf.Session() as sess:
        out = sess.run(l1.built_obj)
    assert out[0, 0, 0] == 25.0
    assert out[0, 0, 1] == 28.0
    assert out[0, 1, 0] == 57.0
    assert out[0, 1, 1] == 64.0
    assert out[0, 2, 0] == 89.0
    assert out[0, 2, 1] == 100.0

    # Single dimension, batch size of 2 (should do each sample independently)
    a = tf.constant([[[1, 2], [3, 4], [5, 6]], 
                     [[2, 3], [4, 5], [6, 7]]],
                    dtype=tf.float32)
    b = tf.constant([[[7, 8], [9, 10]],
                     [[8, 9], [10, 11]]],
                    dtype=tf.float32)
    l1 = Matmul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 2
    assert l1.built_obj.shape[1].value == 3
    assert l1.built_obj.shape[2].value == 2
    with tf.Session() as sess:
        out = sess.run(l1.built_obj)
    assert out[0, 0, 0] == 25.0
    assert out[0, 0, 1] == 28.0
    assert out[0, 1, 0] == 57.0
    assert out[0, 1, 1] == 64.0
    assert out[0, 2, 0] == 89.0
    assert out[0, 2, 1] == 100.0
    assert out[1, 0, 0] == 46.0
    assert out[1, 0, 1] == 51.0
    assert out[1, 1, 0] == 82.0
    assert out[1, 1, 1] == 91.0
    assert out[1, 2, 0] == 118.0
    assert out[1, 2, 1] == 131.0

    # With parameters, 2D
    a = Parameter(shape=[4, 5])
    b = Parameter(shape=[5, 6])
    l1 = Matmul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 4
    assert l1.built_obj.shape[2].value == 6

    # With parameters, >2D (should only do last 2 dimensions)
    a = Parameter(shape=[2, 4, 5])
    b = Parameter(shape=[2, 5, 6])
    l1 = Matmul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [3])
    assert len(l1.built_obj.shape) == 4
    assert l1.built_obj.shape[0].value == 3
    assert l1.built_obj.shape[1].value == 2
    assert l1.built_obj.shape[2].value == 4
    assert l1.built_obj.shape[3].value == 6

    # Should be able to broadcast across the batch!
    a = tf.constant([[[1, 2], [3, 4], [5, 6]], 
                     [[2, 3], [4, 5], [6, 7]]],
                    dtype=tf.float32)
    b = tf.constant([[[7, 8], [9, 10]]], dtype=tf.float32)
    l1 = Matmul(a, b)
    l1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
    assert len(l1.built_obj.shape) == 3
    assert l1.built_obj.shape[0].value == 2
    assert l1.built_obj.shape[1].value == 3
    assert l1.built_obj.shape[2].value == 2
    with tf.Session() as sess:
        out = sess.run(l1.built_obj)
    assert out[0, 0, 0] == 25.0
    assert out[0, 0, 1] == 28.0
    assert out[0, 1, 0] == 57.0
    assert out[0, 1, 1] == 64.0
    assert out[0, 2, 0] == 89.0
    assert out[0, 2, 1] == 100.0
    assert out[1, 0, 0] == 41.0
    assert out[1, 0, 1] == 46.0
    assert out[1, 1, 0] == 73.0
    assert out[1, 1, 1] == 82.0
    assert out[1, 2, 0] == 105.0
    assert out[1, 2, 1] == 118.0
