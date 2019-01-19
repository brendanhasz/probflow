"""Tests importing works correctly."""

import pytest

from types import ModuleType, FunctionType
from inspect import isclass

from probflow import *


def test_modules_import_from_root():
    """Tests that modules import correctly from root."""
    assert isinstance(core, ModuleType)
    assert isinstance(distributions, ModuleType)
    assert isinstance(layers, ModuleType)
    assert isinstance(models, ModuleType)
    assert isinstance(parameters, ModuleType)


def test_core_imports():
    """Tests that stuff in core module imports correctly."""
    assert isinstance(core.REQUIRED, object)
    assert isclass(core.BaseParameter)
    assert isclass(core.BaseLayer)
    assert isclass(core.BaseDistribution)
    assert isclass(core.ContinuousDistribution)
    assert isclass(core.DiscreteDistribution)


def test_parameters_imports():
    """Tests that stuff in parameters module imports correctly."""
    assert isclass(parameters.Parameter)
    assert isinstance(parameters.ScaleParameter, FunctionType)


def test_layers_imports():
    """Tests that stuff in layers module imports correctly."""
    assert isclass(layers.Input)
    assert isclass(layers.Add)
    assert isclass(layers.Sub)
    assert isclass(layers.Mul)
    assert isclass(layers.Div)
    assert isclass(layers.Neg)
    assert isclass(layers.Abs)
    assert isclass(layers.Exp)
    assert isclass(layers.Log)
    assert isclass(layers.Reciprocal)
    assert isclass(layers.Sqrt)
    assert isclass(layers.Sigmoid)
    assert isclass(layers.Relu)
    assert isclass(layers.Softmax)
    assert isclass(layers.Sum)
    assert isclass(layers.Mean)
    assert isclass(layers.Min)
    assert isclass(layers.Max)
    assert isclass(layers.Prod)
    assert isclass(layers.LogSumExp)
    assert isclass(layers.Cat)
    assert isclass(layers.Dot)
    assert isclass(layers.Matmul)
    assert isclass(layers.Dense)
    assert isclass(layers.Sequential)
    assert isclass(layers.Embedding)


def test_distributions_imports():
    """Tests that stuff in distributions module imports correctly."""
    assert isclass(distributions.Normal)
    assert isclass(distributions.HalfNormal)
    assert isclass(distributions.StudentT)
    assert isclass(distributions.Cauchy)
    assert isclass(distributions.Gamma)
    assert isclass(distributions.InvGamma)
    assert isclass(distributions.Bernoulli)
    assert isclass(distributions.Poisson)


def test_models_imports():
    """Tests that stuff in models module imports correctly."""
    assert isinstance(models.LinearRegression, FunctionType)
    assert isinstance(models.LogisticRegression, FunctionType)
    assert isinstance(models.DenseNet, FunctionType)
    assert isinstance(models.DenseRegression, FunctionType)
    assert isinstance(models.DenseClassifier, FunctionType)


def test_submodules_import_from_root():
    """Tests that everything imports from root correctly."""

    # core
    assert isclass(BaseParameter)
    assert isclass(BaseLayer)
    assert isclass(BaseDistribution)
    assert isclass(ContinuousDistribution)
    assert isclass(DiscreteDistribution)

    # parameters
    assert isclass(Parameter)
    assert isinstance(ScaleParameter, FunctionType)

    # layers
    assert isclass(Input)
    assert isclass(Add)
    assert isclass(Sub)
    assert isclass(Mul)
    assert isclass(Div)
    assert isclass(Neg)
    assert isclass(Abs)
    assert isclass(Exp)
    assert isclass(Log)
    assert isclass(Reciprocal)
    assert isclass(Sqrt)
    assert isclass(Sigmoid)
    assert isclass(Relu)
    assert isclass(Softmax)
    assert isclass(Sum)
    assert isclass(Mean)
    assert isclass(Min)
    assert isclass(Max)
    assert isclass(Prod)
    assert isclass(LogSumExp)
    assert isclass(Cat)
    assert isclass(Dot)
    assert isclass(Matmul)
    assert isclass(Dense)
    assert isclass(Sequential)
    assert isclass(Embedding)

    # distributions
    assert isclass(Normal)
    assert isclass(HalfNormal)
    assert isclass(StudentT)
    assert isclass(Cauchy)
    assert isclass(Gamma)
    assert isclass(InvGamma)
    assert isclass(Bernoulli)
    assert isclass(Poisson)

    # models
    assert isinstance(LinearRegression, FunctionType)
    assert isinstance(LogisticRegression, FunctionType)
    assert isinstance(DenseNet, FunctionType)
    assert isinstance(DenseRegression, FunctionType)
    assert isinstance(DenseClassifier, FunctionType)
    assert isinstance(LinearRegression, FunctionType)
    assert isinstance(LinearRegression, FunctionType)
    assert isinstance(LinearRegression, FunctionType)
