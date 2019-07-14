"""Initializers.

Functions to initialize posterior distribution variables.

* :class:`.xavier` - Xavier initializer
* :class:`.scale_xavier` - Xavier initializer scaled for scale parameters
* :class:`.pos_xavier` - positive-only initizlier

----------

"""



from probflow.core.settings import get_backend



def xavier(shape):
    """Xavier initializer

    """
    scale = np.sqrt(2/sum(shape))
    if get_backend() == 'pytorch':
        # TODO: use truncated normal for torch
        return torch.randn(shape) * scale
    else:
        return tf.random.truncated_normal(shape, mean=0.0, stddev=scale)



def scale_xavier(shape):
    """Xavier initializer for scale variables"""
    vals = xavier(shape)
    if get_backend() == 'pytorch':
        numel = torch.prod(shape)
        return vals+2-2*torch.log(numel)/torch.log(10)
    else:
        numel = tf.reduce_prod(shape)
        return vals+2-2*tf.log(numel)/tf.log(10)



def pos_xavier(shape):
    """Xavier initializer for positive variables"""
    vals = xavier(shape)
    if get_backend() == 'pytorch':
        numel = torch.prod(shape)
        return vals + torch.log(numel)/torch.log(10)
    else:
        numel = tf.reduce_prod(shape)
        return vals + tf.log(numel)/tf.log(10)
