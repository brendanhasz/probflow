"""
The core.settings module contains global settings about the backend to use,
what sampling method to use, the default device, and default datatype.


Backend
-------

Which backend to use.  Can be either 
`TensorFlow 2.0 <http://www.tensorflow.org/beta/>`_ 
or `PyTorch <http://pytorch.org/>`_.

* :func:`.get_backend`
* :func:`.set_backend`


Samples
-------

Whether and how many samples to draw from parameter posterior distributions.
If ``None``, the maximum a posteriori estimate of each parameter will be used.
If an integer greater than 0, that many samples from each parameter's posterior
distribution will be used.

* :func:`.get_samples`
* :func:`.set_samples`


Flipout
-------

Whether to use `Flipout <https://arxiv.org/abs/1803.04386>`_ where possible.

* :func:`.get_flipout`
* :func:`.set_flipout`


Sampling context manager
------------------------

A context manager which controls how |Parameters| sample from their 
variational distributions while inside the context manager.

* :class:`.Sampling`


"""


__all__ = [
    'get_backend',
    'set_backend',
    'get_datatype',
    'set_datatype',
    'get_samples',
    'set_samples',
    'get_flipout',
    'set_flipout',
    'Sampling',
]



class _Settings():
    """Class to store ProbFlow global settings

    Attributes
    ----------
    _BACKEND : str {'tensorflow' or 'pytorch'}
        What backend to use
    _SAMPLES : |None| or int > 0
        How many samples to take from |Parameter| variational posteriors.
        If |None|, will use MAP estimates.
    _FLIPOUT : bool
        Whether to use flipout where possible
    _DATATYPE : tf.dtype or ???
        Default datatype to use for tensors
    """

    def __init__(self):
        self._BACKEND = 'tensorflow'
        self._SAMPLES = None
        self._FLIPOUT = False
        self._DATATYPE = None



# Global ProbFlow settings
__SETTINGS__ = _Settings()



def get_backend():
    return __SETTINGS__._BACKEND



def set_backend(backend):
    if isinstance(backend, str):
        if backend in ['tensorflow', 'pytorch']:
            __SETTINGS__._BACKEND = backend
        else:
            raise ValueError('backend must be either tensorflow or pytorch')
    else:
        raise TypeError('backend must be a string')



def get_datatype():
    """Get the datatype to use for Tensors"""
    if __SETTINGS__._DATATYPE is None:
        if get_backend() == 'pytorch':
            import torch
            return torch.float32
        else:
            import tensorflow as tf
            return tf.dtypes.float32
    else:
        return __SETTINGS__._DATATYPE



def set_datatype(datatype):
    """Set the datatype to use for Tensors"""
    if get_backend() == 'pytorch':
        import torch
        if datatype is None or isinstance(datatype, torch.dtype):
            __SETTINGS__._DATATYPE = datatype
        else:
            raise TypeError('datatype must be a torch.dtype')
    else:
        import tensorflow as tf
        if datatype is None or isinstance(datatype, tf.dtypes.DType):
            __SETTINGS__._DATATYPE = datatype
        else:
            raise TypeError('datatype must be a tf.dtypes.DType')


# TODO: default device (for pytorch at least)



def get_samples():
    return __SETTINGS__._SAMPLES



def set_samples(samples):
    if samples is not None and not isinstance(samples, int):
        raise TypeError('samples must be an int or None')
    elif isinstance(samples, int) and samples < 1:
        raise ValueError('samples must be positive')
    else:
        __SETTINGS__._SAMPLES = samples



def get_flipout():
    return __SETTINGS__._FLIPOUT



def set_flipout(flipout):
    if isinstance(flipout, bool):
        __SETTINGS__._FLIPOUT = flipout
    else:
        raise TypeError('flipout must be True or False')



class Sampling():
    """Use sampling while within this context manager."""


    def __init__(self, n=1, flipout=False):
        self._n = n
        self._flipout = flipout


    def __enter__(self):
        """Begin sampling.

        Keyword Arguments
        -----------------
        n : None or int > 0
            Number of samples (if any) to draw from parameters' posteriors.
            Default = 1
        flipout : bool
            Whether to use flipout where possible while sampling.
            Default = False
        """
        set_samples(self._n)
        set_flipout(self._flipout)


    def __exit__(self, _type, _val, _tb):
        """End sampling and reset sampling settings to defaults"""
        set_samples(None)
        set_flipout(False)
