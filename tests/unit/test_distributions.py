"""Tests probflow.distributions modules"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.distributions import *
from probflow.layers import Add
from probflow.parameters import Parameter



def lnormal(mu, sig, x):
    """Log prob of normal dist w/ mean mu, std dev sig, @ x=x"""
    return np.log(1.0 / np.sqrt(2*np.pi*sig*sig) *
                  np.exp(-(x-mu)*(x-mu)/(2*sig*sig)))



def lhnormal(sig, x):
    """Log prob of normal dist w/ mean mu, std dev sig, @ x=x"""
    if x>=0:
        return np.log(2.0 / np.sqrt(2*np.pi*sig*sig) *
                      np.exp(-(x*x)/(2*sig*sig)))
    else:
        return -np.inf



def logit(a):
    return np.log(a) - np.log(1.0-a)



def isclose(a, b, tol=1e-6):
    """Returns true if a and b are w/i tol"""
    return abs(a-b) < tol



def allclose(a, b, tol=1e-6):
    """Returns true if all elements of a and b are w/i tol"""
    return np.all((abs(a-b) < tol) | (a == b) | (np.isnan(a) & np.isnan(b)))



def test_normal_float_input():
    """Tests probflow.distributions.Normal w/ float/int input"""
    d1 = Normal(2.0, 1.0)
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu, 
            sig, 
            lp1, 
            lp2,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(1.0),
            d1.built_obj.log_prob(2.0),
        ])
    assert mu==2.0
    assert sig==1.0
    assert isclose(lp1, lnormal(0, 1, 1))
    assert isclose(lp2, lnormal(0, 1, 0))



def test_normal_numpy_input():
    """Tests probflow.distributions.Normal w/ numpy array input"""
    a = np.array([[0], [2]]).astype('float32')
    b = np.array([[2], [1]]).astype('float32')
    d1 = Normal(a, b)
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu, 
            sig, 
            lp0, 
            lp1,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.array([[0.], [2]])),
            d1.built_obj.log_prob(np.array([[-2.], [3]])),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert isinstance(lp1, np.ndarray)
    assert mu.ndim==2
    assert mu[0][0]==0.0
    assert mu[1][0]==2.0
    assert sig[0][0]==2.0
    assert sig[1][0]==1.0
    assert isclose(lp0[0][0], lnormal(0, 2, 0))
    assert isclose(lp0[1][0], lnormal(0, 1, 0))
    assert isclose(lp1[0][0], lnormal(0, 2, 2))
    assert isclose(lp1[1][0], lnormal(0, 1, 1))



def test_normal_layer_input():
    """Tests probflow.distributions.Normal w/ layer inputs"""
    d1 = Normal(Add(1.0, 1.0), Add(0.5, 0.5))
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu,
            sig,
            lp1,
            lp2,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(1.0),
            d1.built_obj.log_prob(2.0),
        ])
    assert mu==2.0
    assert sig==1.0
    assert isclose(lp1, lnormal(0, 1, 1))
    assert isclose(lp2, lnormal(0, 1, 0))



def test_normal_tensor_input():
    """Tests probflow.distributions.Normal w/ tf.Tensor input"""
    a = tf.constant([[0], [2]], dtype=tf.float32)
    b = tf.constant([[2], [1]], dtype=tf.float32)
    d1 = Normal(a, b)
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu, 
            sig, 
            lp0, 
            lp1,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.array([[0.], [2]])),
            d1.built_obj.log_prob(np.array([[-2.], [3]])),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert isinstance(lp1, np.ndarray)
    assert mu.ndim==2
    assert mu[0][0]==0.0
    assert mu[1][0]==2.0
    assert sig[0][0]==2.0
    assert sig[1][0]==1.0
    assert isclose(lp0[0][0], lnormal(0, 2, 0))
    assert isclose(lp0[1][0], lnormal(0, 1, 0))
    assert isclose(lp1[0][0], lnormal(0, 2, 2))
    assert isclose(lp1[1][0], lnormal(0, 1, 1))



def test_normal_variable_input():
    """Tests probflow.distributions.Normal w/ tf.Variable input"""
    a = tf.Variable([[0], [2]], dtype=tf.float32)
    b = tf.Variable([[2], [1]], dtype=tf.float32)
    d1 = Normal(a, b)
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(d1.built_obj, tfd.Normal)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [
            mu, 
            sig, 
            lp0, 
            lp1,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.array([[0.], [2]])),
            d1.built_obj.log_prob(np.array([[-2.], [3]])),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert isinstance(lp1, np.ndarray)
    assert mu.ndim==2
    assert mu[0][0]==0.0
    assert mu[1][0]==2.0
    assert sig[0][0]==2.0
    assert sig[1][0]==1.0
    assert isclose(lp0[0][0], lnormal(0, 2, 0))
    assert isclose(lp0[1][0], lnormal(0, 1, 0))
    assert isclose(lp1[0][0], lnormal(0, 2, 2))
    assert isclose(lp1[1][0], lnormal(0, 1, 1))



def test_normal_parameter_input():
    """Tests probflow.distributions.Normal w/ Parameter input"""
    a = Parameter(shape=[3,4])
    b = Parameter(shape=[3,4])
    d1 = Normal(a, b)
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [5])
    assert isinstance(d1.built_obj, tfd.Normal)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [
            mu, 
            sig, 
            lp0, 
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.random.randn(5, 3, 4)),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert mu.ndim == 3
    assert mu.shape[0] == 5 # 1st dim should be batch size w/ Parameter!
    assert mu.shape[1] == 3
    assert mu.shape[2] == 4
    assert sig.ndim == 3
    assert sig.shape[0] == 5
    assert sig.shape[1] == 3
    assert sig.shape[2] == 4
    assert lp0.ndim == 3
    assert lp0.shape[0] == 5
    assert lp0.shape[1] == 3
    assert lp0.shape[2] == 4



def _test_dist(dist, tfd_dist, params, x_eval, lp_eval):
    """Test an arbitrary distribution"""

    # Convert outputs to numpy arrays
    x_eval = np.array(x_eval, dtype='float32')
    lp_eval = np.array(lp_eval, dtype='float32')

    # Convert inputs to tensors
    tensor_in = [tf.constant(p, dtype=tf.float32) for p in params]

    # Build the distribution
    d1 = dist(*tensor_in)
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(d1.built_obj, tfd_dist)

    # Check the outputs are correct
    with tf.Session() as sess:
        [
            lp
        ] = sess.run([
            d1.built_obj.log_prob(x_eval)
        ])
    assert isinstance(lp, np.ndarray)
    assert lp.ndim == lp_eval.ndim
    assert lp.shape == lp_eval.shape
    assert allclose(lp, lp_eval)



def test_halfnormal():
    """Tests probflow.distributions.HalfNormal"""

    sigma = [[1], [2]]

    x = [[0], [0]]
    lp = [[lhnormal(1, 0)], [lhnormal(2, 0)]]
    _test_dist(HalfNormal, tfd.HalfNormal, [sigma], x, lp)

    x = [[1], [2]]
    lp = [[lhnormal(1, 1)], [lhnormal(2, 2)]]
    _test_dist(HalfNormal, tfd.HalfNormal, [sigma], x, lp)

    x = [[-2], [-1]]
    lp = [[-np.inf], [-np.inf]]
    _test_dist(HalfNormal, tfd.HalfNormal, [sigma], x, lp)



def test_deterministic():
    """Tests probflow.distributions.Deterministic"""

    loc = [[1], [2]]

    # log prob should be 0 (raw prob=1) when value matches loc, -inf elsewhere
    x = [[1], [0]]
    lp = [[0], [-np.inf]]
    _test_dist(Deterministic, tfd.Deterministic, [loc], x, lp)

    x = [[1.01], [2]]
    lp = [[-np.inf], [0]]
    _test_dist(Deterministic, tfd.Deterministic, [loc], x, lp)



def test_studentt():
    """Tests probflow.distributions.StudentT"""

    df = [[1], [2]]
    loc = [[0], [-1]]
    scale = [[0.5], [1]]

    x = [[1], [0]]
    lp = [[-2.061020617723555], 
          [-1.6479184330021643]]
    _test_dist(StudentT, tfd.StudentT, [df, loc, scale], x, lp)

    x = [[-0.5], [1.5]]
    lp = [[-1.1447298858494], 
          [-3.1653197]]
    _test_dist(StudentT, tfd.StudentT, [df, loc, scale], x, lp)



def test_cauchy():
    """Tests probflow.distributions.Cauchy"""

    loc = [[0], [-1]]
    scale = [[0.5], [1]]

    x = [[1], [0]]
    lp = [[-2.061020618], 
          [-1.837877066]]
    _test_dist(Cauchy, tfd.Cauchy, [loc, scale], x, lp)

    x = [[-0.5], [1.5]]
    lp = [[-1.144729886], 
          [-3.125731354]]
    _test_dist(Cauchy, tfd.Cauchy, [loc, scale], x, lp)



def test_gamma():
    """Tests probflow.distributions.Gamma"""

    shape = [[1], [5]]
    rate = [[0.5], [1]]

    x = [[2], [4]]
    lp = [[-1.693147181], 
          [-1.632876386]]
    _test_dist(Gamma, tfd.Gamma, [shape, rate], x, lp)

    x = [[4], [2]]
    lp = [[-2.693147181], 
          [-2.405465108]]
    _test_dist(Gamma, tfd.Gamma, [shape, rate], x, lp)



def test_invgamma():
    """Tests probflow.distributions.InvGamma"""

    shape = [[1], [3]]
    rate = [[1], [0.5]]

    x = [[0.5], [0.25]]
    lp = [[-0.6137056389], 
          [0.7725887222]]
    _test_dist(InvGamma, tfd.InverseGamma, [shape, rate], x, lp)

    x = [[0.25], [0.5]]
    lp = [[-1.227411278], 
          [-1.0]]
    _test_dist(InvGamma, tfd.InverseGamma, [shape, rate], x, lp)



def test_poisson():
    """Tests probflow.distributions.Poisson"""

    rate = [[1], [10]]

    x = [[2], [5]]
    lp = [[-1.693147181], 
          [-3.274566278]]
    _test_dist(Poisson, tfd.Poisson, [rate], x, lp)

    x = [[5], [2]]
    lp = [[-5.787491743], 
          [-6.087976995]]
    _test_dist(Poisson, tfd.Poisson, [rate], x, lp)



def test_bernoulli():
    """Tests probflow.distributions.Bernoulli"""

    # Test w/ input_type=probs
    prob = [[0.5], [0.1], [0.1]]
    x = [[0], [0], [1]]
    prob_out = [[np.log(0.5)], [np.log(0.9)], [np.log(0.1)]]
    the_dist = lambda v: Bernoulli(v, input_type='probs')
    _test_dist(the_dist, tfd.Bernoulli, [prob], x, prob_out)

    # Test w/ input_type=logits
    prob = [[logit(0.5)], [logit(0.1)], [logit(0.1)]]
    x = [[0], [0], [1]]
    prob_out = [[np.log(0.5)], [np.log(0.9)], [np.log(0.1)]]
    the_dist = lambda v: Bernoulli(v, input_type='logits')
    _test_dist(the_dist, tfd.Bernoulli, [prob], x, prob_out)

    # Default should be input_type=logits
    prob = [[logit(0.5)], [logit(0.1)], [logit(0.1)]]
    x = [[0], [0], [1]]
    prob_out = [[np.log(0.5)], [np.log(0.9)], [np.log(0.1)]]
    _test_dist(Bernoulli, tfd.Bernoulli, [prob], x, prob_out)



def test_categorical():
    """Tests probflow.distributions.Categorical"""

    # Test w/ input_type=probs
    probs_in = np.array([[0.3, 0.2, 0.5], [0.7, 0.2, 0.1]])
    x_eval = np.array([0, 1])
    d1 = Categorical(probs_in, input_type='probs')
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
    assert isinstance(d1.built_obj, tfd.Categorical)
    with tf.Session() as sess:
        [
            rp,
            lp
        ] = sess.run([
            d1.built_obj.prob(x_eval),
            d1.built_obj.log_prob(x_eval)
        ])
    assert isinstance(rp, np.ndarray)
    assert isinstance(lp, np.ndarray)
    assert rp.ndim == 1
    assert lp.ndim == 1
    assert rp.shape[0] == 2
    assert lp.shape[0] == 2
    assert isclose(rp[0], 0.3)
    assert isclose(rp[1], 0.2)
    assert isclose(lp[0], np.log(0.3))
    assert isclose(lp[1], np.log(0.2))

    # Test w/ input_type=logits
    probs_in = np.log(np.array([[0.3, 0.2, 0.5], [0.7, 0.2, 0.1]]))
    x_eval = np.array([0, 1])
    d1 = Categorical(probs_in, input_type='logits')
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
    assert isinstance(d1.built_obj, tfd.Categorical)
    with tf.Session() as sess:
        [
            rp,
            lp
        ] = sess.run([
            d1.built_obj.prob(x_eval),
            d1.built_obj.log_prob(x_eval)
        ])
    assert isinstance(rp, np.ndarray)
    assert isinstance(lp, np.ndarray)
    assert rp.ndim == 1
    assert lp.ndim == 1
    assert rp.shape[0] == 2
    assert lp.shape[0] == 2
    assert isclose(rp[0], 0.3)
    assert isclose(rp[1], 0.2)
    assert isclose(lp[0], np.log(0.3))
    assert isclose(lp[1], np.log(0.2))

    # Test w/ input_type=raw and input as a numpy array
    probs_in = np.log(np.array([[0.3, 0.5], 
                                [0.7, 0.3], 
                                [0.7, 0.3], 
                                [0.7, 0.3]]))
    x_eval = np.array([0, 0, 1, 2])
    d1 = Categorical(probs_in, input_type='raw')
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [4])
    assert isinstance(d1.built_obj, tfd.Categorical)
    with tf.Session() as sess:
        [
            rp,
            lp
        ] = sess.run([
            d1.built_obj.prob(x_eval),
            d1.built_obj.log_prob(x_eval)
        ])
    assert isinstance(rp, np.ndarray)
    assert isinstance(lp, np.ndarray)
    assert rp.ndim == 1
    assert lp.ndim == 1
    assert rp.shape[0] == 4
    assert lp.shape[0] == 4
    assert isclose(rp[0], 0.3/(0.3+0.5+1))
    assert isclose(rp[1], 0.7/(0.7+0.3+1))
    assert isclose(rp[2], 0.3/(0.7+0.3+1))
    assert isclose(rp[3], 1.0/(0.7+0.3+1))
    assert isclose(lp[0], np.log(0.3/(0.3+0.5+1)))
    assert isclose(lp[1], np.log(0.7/(0.7+0.3+1)))
    assert isclose(lp[2], np.log(0.3/(0.7+0.3+1)))
    assert isclose(lp[3], np.log(1.0/(0.7+0.3+1)))

    # Test w/ input_type=raw and input as a TF tensor
    probs_in = tf.constant(np.log(np.array([[0.3, 0.5], 
                                            [0.7, 0.3], 
                                            [0.7, 0.3], 
                                            [0.7, 0.3]])), dtype=tf.float32)
    x_eval = np.array([0, 0, 1, 2])
    d1 = Categorical(probs_in, input_type='raw')
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [4])
    assert isinstance(d1.built_obj, tfd.Categorical)
    with tf.Session() as sess:
        [
            rp,
            lp
        ] = sess.run([
            d1.built_obj.prob(x_eval),
            d1.built_obj.log_prob(x_eval)
        ])
    assert isinstance(rp, np.ndarray)
    assert isinstance(lp, np.ndarray)
    assert rp.ndim == 1
    assert lp.ndim == 1
    assert rp.shape[0] == 4
    assert lp.shape[0] == 4
    assert isclose(rp[0], 0.3/(0.3+0.5+1))
    assert isclose(rp[1], 0.7/(0.7+0.3+1))
    assert isclose(rp[2], 0.3/(0.7+0.3+1))
    assert isclose(rp[3], 1.0/(0.7+0.3+1))
    assert isclose(lp[0], np.log(0.3/(0.3+0.5+1)))
    assert isclose(lp[1], np.log(0.7/(0.7+0.3+1)))
    assert isclose(lp[2], np.log(0.3/(0.7+0.3+1)))
    assert isclose(lp[3], np.log(1.0/(0.7+0.3+1)))

    # Test w/ input_type=raw and input as a parameter
    probs_in = Parameter(shape=(4, 2))
    x_eval = np.array([[0, 1, 2, 1],
                       [0, 1, 2, 1],
                       [0, 1, 2, 1],
                       [0, 1, 2, 1],
                       [0, 1, 2, 1]])
    d1 = Categorical(probs_in, input_type='raw')
    d1._build_recursively(tf.placeholder(tf.float32, [1]), [5])
    assert isinstance(d1.built_obj, tfd.Categorical)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [
            rp,
        ] = sess.run([
            d1.built_obj.prob(x_eval),
        ])
    assert isinstance(rp, np.ndarray)
    assert rp.ndim == 2
    assert rp.shape[0] == 5
    assert rp.shape[1] == 4
