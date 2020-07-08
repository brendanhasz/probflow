import os
import uuid
import numpy as np
import probflow as pf


def isclose(a, b, thresh=1e-6):
    return np.all(np.abs(a-b) < thresh)


def get_test_data(N, D):
    x = np.random.randn(N, D).astype('float32')
    w = np.random.randn(D, 1).astype('float32')
    y = x@w + 0.1*np.random.randn(N, 1).astype('float32')
    return x, y


def test_dumps_and_loads_before_fitting():
    model1 = pf.LinearRegression(7)
    model2 = pf.loads(model1.dumps())
    assert isinstance(model2, pf.LinearRegression)
    assert model1 is not model2
    post1 = model1.posterior_mean()
    post2 = model2.posterior_mean()
    for k in post1:
        assert isclose(post1[k], post2[k])


def test_dump_and_load_before_fitting(tmpdir):
    model1 = pf.LinearRegression(7)
    fname = str(tmpdir.join('test_model.pkl'))
    model1.save(fname)
    model2 = pf.load(fname)
    assert isinstance(model2, pf.LinearRegression)
    assert model1 is not model2
    post1 = model1.posterior_mean()
    post2 = model2.posterior_mean()
    for k in post1:
        assert isclose(post1[k], post2[k])


def test_dumps_and_loads_after_fitting():
    model1 = pf.LinearRegression(7)
    x, y = get_test_data(1024, 7)
    model1.fit(x, y, epochs=2)
    model2 = pf.loads(model1.dumps())
    assert isinstance(model2, pf.LinearRegression)
    assert model1 is not model2
    post1 = model1.posterior_mean()
    post2 = model2.posterior_mean()
    for k in post1:
        assert isclose(post1[k], post2[k])


def test_dump_and_load_after_fitting(tmpdir):
    model1 = pf.LinearRegression(7)
    x, y = get_test_data(1024, 7)
    model1.fit(x, y, epochs=2)
    fname = str(tmpdir.join('test_model.pkl'))
    model1.save(fname)
    model2 = pf.load(fname)
    assert isinstance(model2, pf.LinearRegression)
    assert model1 is not model2
    post1 = model1.posterior_mean()
    post2 = model2.posterior_mean()
    for k in post1:
        assert isclose(post1[k], post2[k])
