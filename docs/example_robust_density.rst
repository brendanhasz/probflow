.. _example_robust_density:

Robust Density Network
======================

.. include:: macros.hrst


TODO: dual-module net which estimates predictions and uncertainty separately, and uses a t-dist for the observation dist

TODO: math and diagram

.. code-block:: python3

    class RobustDensityNetwork(pf.Model):
        
        def __init__(self, units, head_units):
            self.core_net = pf.DenseNetwork(units)
            self.loc_net = pf.DenseNetwork([units[-1]]+head_units)
            self.std_net = pf.DenseNetwork([units[-1]]+head_units)
            
        def __call__(self, x):
            x = tf.nn.relu(self.core_net(x))
            loc = self.loc_net(x)
            std = tf.nn.softplus(self.std_net(x))
            return pf.StudentT(loc, std)