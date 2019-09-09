"""Tests the probflow.distributions module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import probflow.distributions as pfd



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_Deterministic():
    """Tests Deterministic distribution"""

    # Create the distribution
    dist = pfd.Deterministic()

    # Check default params
    assert dist.loc == 0

    # Call should return backend obj
    assert isinstance(dist(), tfd.Deterministic)

    # Test methods
    assert dist.prob(0).numpy() == 1.0
    assert dist.prob(1).numpy() == 0.0
    assert dist.log_prob(0).numpy() == 0.0
    assert dist.log_prob(1).numpy() == -np.inf
    assert dist.mean().numpy() == 0.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.Deterministic(loc=3)
    assert dist.loc == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Deterministic(loc='lalala')



def test_Normal():
    """Tests Normal distribution"""

    # Create the distribution
    dist = pfd.Normal()

    # Check default params
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert isinstance(dist(), tfd.Normal)

    # Test methods
    npdf = lambda x, m, s: (1.0/np.sqrt(2*np.pi*s*s) * 
    						np.exp(-np.power(x-m, 2)/(2*s*s)))
    assert is_close(dist.prob(0).numpy(), npdf(0, 0, 1))
    assert is_close(dist.prob(1).numpy(), npdf(1, 0, 1))
    assert is_close(dist.log_prob(0).numpy(), np.log(npdf(0, 0, 1)))
    assert is_close(dist.log_prob(1).numpy(), np.log(npdf(1, 0, 1)))
    assert dist.mean().numpy() == 0.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.Normal(loc=3, scale=2)
    assert dist.loc == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Normal(loc='lalala', scale='lalala')



def test_StudentT():
    """Tests StudentT distribution"""

    # Create the distribution
    dist = pfd.StudentT()

    # Check default params
    assert dist.df == 1
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert isinstance(dist(), tfd.StudentT)

    # Test methods
    cpdf = lambda x, m, s: 1.0/(np.pi*s*(1+(np.power((x-m)/s, 2))))
    assert is_close(dist.prob(0).numpy(), cpdf(0, 0, 1))
    assert is_close(dist.prob(1).numpy(), cpdf(1, 0, 1))
    assert is_close(dist.log_prob(0).numpy(), np.log(cpdf(0, 0, 1)))
    assert is_close(dist.log_prob(1).numpy(), np.log(cpdf(1, 0, 1)))
    assert dist.mean() == 0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.StudentT(df=5, loc=3, scale=2)
    assert dist.df == 5
    assert dist.loc == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.StudentT(df='lalala', loc='lalala', scale='lalala')



def test_Cauchy():
    """Tests Cauchy distribution"""

    # Create the distribution
    dist = pfd.Cauchy()

    # Check default params
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert isinstance(dist(), tfd.Cauchy)

    # Test methods
    cpdf = lambda x, m, s: 1.0/(np.pi*s*(1+(np.power((x-m)/s, 2))))
    assert is_close(dist.prob(0).numpy(), cpdf(0, 0, 1))
    assert is_close(dist.prob(1).numpy(), cpdf(1, 0, 1))
    assert is_close(dist.log_prob(0).numpy(), np.log(cpdf(0, 0, 1)))
    assert is_close(dist.log_prob(1).numpy(), np.log(cpdf(1, 0, 1)))
    assert dist.mean() == 0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.Cauchy(loc=3, scale=2)
    assert dist.loc == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Cauchy(loc='lalala', scale='lalala')



def test_Gamma():
    """Tests Gamma distribution"""

    # Create the distribution
    dist = pfd.Gamma(5, 4)

    # Check default params
    assert dist.concentration == 5
    assert dist.rate == 4

    # Call should return backend obj
    assert isinstance(dist(), tfd.Gamma)

    # Test methods
    assert is_close(dist.prob(0).numpy(), 0.0)
    assert is_close(dist.prob(1).numpy(), 0.78146726)
    assert dist.log_prob(0).numpy() == -np.inf
    assert is_close(dist.log_prob(1).numpy(), np.log(0.78146726))
    assert is_close(dist.mean(), 5.0/4.0)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.Gamma(3, 2)
    assert dist.concentration == 3
    assert dist.rate == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Gamma('lalala', 'lalala')



def test_InverseGamma():
    """Tests InverseGamma distribution"""

    # Create the distribution
    dist = pfd.InverseGamma(5, 4)

    # Check default params
    assert dist.concentration == 5
    assert dist.scale == 4

    # Call should return backend obj
    assert isinstance(dist(), tfd.InverseGamma)

    # Test methods
    assert is_close(dist.prob(1).numpy(), 0.78146726)
    assert is_close(dist.prob(2).numpy(), 0.09022352)
    assert is_close(dist.log_prob(1).numpy(), np.log(0.78146726))
    assert is_close(dist.log_prob(2).numpy(), np.log(0.09022352))
    assert dist.mean().numpy() == 1.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.InverseGamma(3, 2)
    assert dist.concentration == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.InverseGamma('lalala', 'lalala')



def test_Bernoulli():
    """Tests Bernoulli distribution"""

    # Create the distribution
    dist = pfd.Bernoulli(0)

    # Check default params
    assert dist.logits == 0
    assert dist.probs is None

    # Call should return backend obj
    assert isinstance(dist(), tfd.Bernoulli)

    # Test methods
    assert is_close(dist.prob(0).numpy(), 0.5)
    assert is_close(dist.prob(1).numpy(), 0.5)
    assert is_close(dist.log_prob(0).numpy(), np.log(0.5))
    assert is_close(dist.log_prob(1).numpy(), np.log(0.5))
    assert dist.mean().numpy() == 0.5

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.Bernoulli(probs=0.8)
    assert dist.probs == 0.8
    assert dist.logits is None
    assert is_close(dist.prob(0).numpy(), 0.2)
    assert is_close(dist.prob(1).numpy(), 0.8)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Bernoulli('lalala')



def test_Categorical():
    """Tests Categorical distribution"""

    # Create the distribution
    dist = pfd.Categorical([0, 1, 2])

    # Check default params
    assert dist.logits == [0, 1, 2]
    assert dist.probs is None

    # Call should return backend obj
    assert isinstance(dist(), tfd.Categorical)

    # Test methods
    assert dist.prob(0).numpy() < dist.prob(1).numpy()
    assert dist.prob(1).numpy() < dist.prob(2).numpy()
    assert dist.log_prob(0).numpy() < dist.log_prob(1).numpy()
    assert dist.log_prob(1).numpy() < dist.log_prob(2).numpy()

    # Mean should return the mode!
    assert dist.mean().numpy() == 2

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = pfd.Categorical(probs=[0.1, 0.7, 0.2])
    assert dist.probs == [0.1, 0.7, 0.2]
    assert dist.logits is None
    assert is_close(dist.prob(0).numpy(), 0.1)
    assert is_close(dist.prob(1).numpy(), 0.7)
    assert is_close(dist.prob(2).numpy(), 0.2)
    assert dist.mean().numpy() == 1

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Categorical('lalala')

	# Should use the last dim if passed a Tensor arg
    dist = pfd.Categorical(probs=[[0.1, 0.7, 0.2], 
    							  [0.8, 0.1, 0.1], 
    	                          [0.01, 0.01, 0.98],
    	                          [0.3, 0.3, 0.4]])
    assert is_close(dist.prob([0, 1, 2, 2]).numpy()[0], 0.1)
    assert is_close(dist.prob([0, 1, 2, 2]).numpy()[1], 0.1)
    assert is_close(dist.prob([0, 1, 2, 2]).numpy()[2], 0.98)
    assert is_close(dist.prob([0, 1, 2, 2]).numpy()[3], 0.4)
    assert is_close(dist.prob([2, 1, 0, 0]).numpy()[0], 0.2)
    assert is_close(dist.prob([2, 1, 0, 0]).numpy()[1], 0.1)
    assert is_close(dist.prob([2, 1, 0, 0]).numpy()[2], 0.01)
    assert is_close(dist.prob([2, 1, 0, 0]).numpy()[3], 0.3)

    # And ensure sample dims are correct
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 4
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4



def test_Poisson():
    """Tests Poisson distribution"""

    # Create the distribution
    dist = pfd.Poisson(3)

    # Check default params
    assert dist.rate == 3

    # Call should return backend obj
    assert isinstance(dist(), tfd.Poisson)

    # Test methods
    ppdf = lambda x, r: np.power(r, x)*np.exp(-r)/np.math.factorial(x)
    assert is_close(dist.prob(0).numpy(), ppdf(0, 3))
    assert is_close(dist.prob(1).numpy(), ppdf(1, 3))
    assert is_close(dist.prob(2).numpy(), ppdf(2, 3))
    assert is_close(dist.prob(3).numpy(), ppdf(3, 3))
    assert is_close(dist.log_prob(0).numpy(), np.log(ppdf(0, 3)))
    assert is_close(dist.log_prob(1).numpy(), np.log(ppdf(1, 3)))
    assert is_close(dist.log_prob(2).numpy(), np.log(ppdf(2, 3)))
    assert is_close(dist.log_prob(3).numpy(), np.log(ppdf(3, 3)))
    assert dist.mean().numpy() == 3

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Poisson('lalala')



def test_Dirichlet():
    """Tests Dirichlet distribution"""

    # Create the distribution
    dist = pfd.Dirichlet([1, 2, 3])

    # Check default params
    assert dist.concentration == [1, 2, 3]

    # Call should return backend obj
    assert isinstance(dist(), tfd.Dirichlet)


    # Test methods
    assert is_close(dist.prob([0, 0, 1]).numpy(), 0.0)
    assert is_close(dist.prob([0, 1, 0]).numpy(), 0.0)
    assert is_close(dist.prob([1, 0, 0]).numpy(), 0.0)
    assert is_close(dist.prob([0.3, 0.3, 0.4]).numpy(), 2.88)
    assert dist.log_prob([0, 0, 1]).numpy() == -np.inf
    assert is_close(dist.log_prob([0.3, 0.3, 0.4]).numpy(), np.log(2.88))
    assert is_close(dist.mean().numpy()[0], 1.0/6.0)
    assert is_close(dist.mean().numpy()[1], 2.0/6.0)
    assert is_close(dist.mean().numpy()[2], 3.0/6.0)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 3
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
	    dist = pfd.Dirichlet('lalala')

	# Should use the last dim if passed a Tensor arg
    dist = pfd.Dirichlet([[1, 2, 3],
    					  [3, 2, 1],
    					  [1, 1, 1],
    					  [100, 100, 100]])
    probs = dist.prob([[0, 0, 1],
    	 			   [1, 0, 0],
    	 			   [0.2, 0.2, 0.6],
    	 			   [1.0/3.0, 1.0/3.0, 1.0/3.0]]).numpy()
    assert probs.ndim == 1
    assert is_close(probs[0], 0.0)
    assert is_close(probs[1], 0.0)
    assert is_close(probs[2], 2.0)
    assert probs[3] > 100.0

    # And ensure sample dims are correct
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 4
    assert samples.shape[1] == 3
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 3
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
    assert samples.shape[2] == 3
