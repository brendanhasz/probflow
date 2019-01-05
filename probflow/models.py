"""Common already-made models.

TODO: more info...

* :class:`.LinearRegression`
* :class:`.LogisticRegression`
* :class:`.DenseRegression`
* :class:`.DenseClassifier`

----------

"""


from .core import ContinuousModel, CategoricalModel

    

class LinearRegression(ContinuousModel):
    """Linear regression model.
    """
    #TODO
    pass


class LogisticRegression(CategoricalModel):
    """Logistic regression model.
    """
    #TODO
    pass


class DenseRegression(ContinuousModel):
    """Regression model using a densely-connected multi-layer neural network.
    """
    #TODO
    pass


class DenseClassifier(CategoricalModel):
    """Classifier model using a densely-connected multi-layer neural network.
    """
    #TODO    
    pass


class Conv1dRegression(ContinuousModel):
    """TODO
    """
    #TODO    
    pass


class Conv1dClassifier(CategoricalModel):
    """TODO
    """
    #TODO    
    pass


class Conv2dRegression(ContinuousModel):
    """TODO
    """
    #TODO    
    pass


class Conv2dClassifier(CategoricalModel):
    """TODO
    """
    #TODO    
    pass


class DenseAutoencoderRegression(ContinuousModel):
    """TODO
    """
    #TODO    
    pass


class DenseAutoencoderClassifier(CategoricalModel):
    """TODO
    """
    #TODO    
    pass


class Conv1dAutoencoderRegression(ContinuousModel):
    """TODO
    """
    #TODO    
    pass


class Conv1dAutoencoderClassifier(CategoricalModel):
    """TODO
    """
    #TODO    
    pass


class Conv2dAutoencoderRegression(ContinuousModel):
    """TODO
    """
    #TODO    
    pass


class Conv2dAutoencoderClassifier(CategoricalModel):
    """TODO
    """
    #TODO    
    pass


#TODO: NeuralMatrixFactorization

#TODO: BayesianCorrelation