.. _example_multilevel:

Mixed Effects and Multilevel Models
===================================

.. include:: macros.hrst

TODO

Basic multilevel model:

.. math::
   
   \beta_s \sim \mathcal{N}(0, \sigma^2_\beta), ~ s = 1, ... , N
   
   y_{is} \sim \mathcal{N}(\mu + \beta_s, \sigma^2), ~ i = 1, ..., n_s, ~ s = 1, ... , N

Can be fit using probflow by:

.. code-block:: python

    from probflow import Parameter, ScaleParameter, Normal

    # df = DataFrame with 2 columns: 'group' and 'observation'
    N = df['group'].nunique()

    G = Input('Group')
    
    pop_mean = Parameter()
    pop_std = ScaleParameter()
    data_std = ScaleParameter()
    
    beta = Parameter(shape=N, prior=Normal(0, pop_std))
    model = Normal(pop_mean + beta[G], data_std)
