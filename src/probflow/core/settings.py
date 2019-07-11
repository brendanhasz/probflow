"""Global settings

The core.settings module contains global settings about the backend to use,
what sampling method to use, the default device, and default datatype.

"""


__all__ = [
    'get_backend',
    'set_backend',
    'get_sampling',
    'set_sampling',
    'get_flipout',
    'set_flipout',
    'Sampling',
]



# What backend to use
_BACKEND = 'tensorflow' #or pytorch



# Whether to sample from Parameter posteriors or use MAP estimates
_SAMPLING = False



# Whether to use flipout where possible
_FLIPOUT = True



def get_backend():
    return _BACKEND



def set_backend(backend):
    if isinstance(backend, str):
        if backend in ['tensorflow', 'pytorch']:
            _BACKEND = backend
        else:
            raise ValueError('backend must be either tensorflow or pytorch')
    else:
        raise ValueError('backend must be a string')



def get_sampling():
    return _SAMPLING



def set_sampling(sampling):
    if isinstance(sampling, bool):
        _SAMPLING = sampling
    else:
        raise TypeError('sampling must be True or False')



def get_flipout():
    return _SAMPLING



def set_flipout(flipout):
    if isinstance(flipout, bool):
        _FLIPOUT = flipout
    else:
        raise TypeError('flipout must be True or False')



class Sampling():
    """Use sampling while within this context manager"""


    def __enter__(self):
        set_sampling(True)


    def __exit__(self, _type, _val, _tb):
        set_sampling(False)


# TODO might have to have a way to set the NUMBER of samples to draw 
# simultaneously

# TODO also setting sampling flag might be a problem when using @tf.function
# or pytorch jit/@script?
