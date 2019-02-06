"""Tests probflow.core.BaseLayer class"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *


def test_BaseLayer_build(LR1_novar_unfit):
    """Tests core.BaseLayer.build"""

    # Build model
    model = LR1_novar_unfit #fixture from conftest.py
    model.build(tf.placeholder(tf.float32, [1]), [1])

    # Check parameter list
    assert isinstance(model._parameters, list)

    # Check args
    assert isinstance(model.built_args, dict)
    assert isinstance(model.mean_args, dict)
    assert 'loc' in model.built_args and 'scale' in model.built_args
    assert 'loc' in model.mean_args and 'scale' in model.mean_args
    assert isinstance(model.built_args['loc'], tf.Tensor)
    assert isinstance(model.built_args['scale'], float)
    assert isinstance(model.mean_args['loc'], tf.Tensor)
    assert isinstance(model.mean_args['scale'], float)

    # Check losses
    assert isinstance(model.samp_loss_sum, tf.Tensor)
    assert isinstance(model.mean_loss_sum, tf.Tensor)
    assert isinstance(model.kl_loss_sum, tf.Tensor)

    # Check built model object
    assert isinstance(model.built_obj, tfd.Normal)
    assert isinstance(model.mean_obj, tfd.Normal)



def test_BaseLayer_parameter_list(LR1_novar_unfit):
    """Tests core.BaseLayer._parameter_list"""

    model = LR1_novar_unfit #fixture from conftest.py

    # Get the parameter list
    params = LR1_novar_unfit._parameter_list() 
    assert isinstance(params, list)
    assert len(params) == 2
    assert all([isinstance(p, BaseParameter) for p in params])
    names = [p.name for p in params]
    assert 'thing1' in names
    assert 'thing2' in names


def test_BaseLayer_str(LR1_novar_unfit):
    """Tests core.BaseLayer.__str__"""
    #weight = Parameter(name='thing_red')
    #bias = Parameter(name='thing_blue')
    #data = Input()
    #model = Normal(data*weight + bias, 3.0)
    model = LR1_novar_unfit #fixture from conftest.py
    real_name = ('Normal\n  loc = \n    Add\n      a = \n        Mul' + 
                 '\n          a = Input (all columns)\n          b = ' +
                 '\n            Parameter \'thing1\' shape=(1,) ' +
                 'prior=Normal(loc=0,scale=1) posterior=Normal\n      b = ' +
                 '\n        Parameter \'thing2\' shape=(1,) ' +
                 'prior=Normal(loc=0,scale=1) posterior=Normal\n  scale = 1.0')
    assert model.__str__() == real_name
