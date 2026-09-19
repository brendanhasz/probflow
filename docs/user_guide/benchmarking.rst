.. _user_guide_benchmarking:

Benchmarking
============

.. include:: ../macros.hrst

ProbFlow's benchmarking suite fits a Bayesian linear regression model (:class:`.LinearRegression`) with each supported backend (TensorFlow, PyTorch, and JAX), for a range of dataset sizes with 100 dimensions.  For each combination it measures the time to train the model, the time to generate predictions, and the time to draw samples from the model's predictive distribution.  Training is additionally benchmarked in both eager and non-eager (compiled) modes, for smaller datasets (for larger datasets non-eager/compiled mode is always used).

Eager vs compiled
--------------------------------------

The plot below compares training runtime in eager vs non-eager (compiled) mode for each backend, using the smallest number of datapoints and the largest number of dimensions benchmarked.

.. image:: ../img/benchmarking/eager_vs_noneager.png
   :width: 70 %
   :align: center

Performance by backend type
---------------------------

The plots below show runtime as a function of the number of datapoints, at the largest number of dimensions benchmarked, with a separate line for each backend.  Only non-eager (compiled) training runs are included.

.. tabs::

    .. group-tab:: Train

        .. image:: ../img/benchmarking/backend_comparison_train.png
           :width: 70 %
           :align: center

    .. group-tab:: Predict

        .. image:: ../img/benchmarking/backend_comparison_predict.png
           :width: 70 %
           :align: center

    .. group-tab:: Sample

        .. image:: ../img/benchmarking/backend_comparison_sample.png
           :width: 70 %
           :align: center


Performance on CPU vs GPU
-------------------------

The plots below show training runtime as a function of the number of datapoints (both on log scales), comparing CPU and GPU execution for each backend.  Only non-eager (compiled) training runs are included.

.. tabs::

    .. group-tab:: Jax

        .. image:: ../img/benchmarking/cpu_vs_gpu_jax.png
           :width: 70 %
           :align: center

    .. group-tab:: Pytorch

        .. image:: ../img/benchmarking/cpu_vs_gpu_pytorch.png
           :width: 70 %
           :align: center

    .. group-tab:: Tensorflow

        .. image:: ../img/benchmarking/cpu_vs_gpu_tensorflow.png
           :width: 70 %
           :align: center


Full benchmarking results
-------------------------

The full set of benchmarking results:

.. list-table::
   :header-rows: 1

   * - operation
     - n_datapoints
     - eager
     - backend
     - runtime_seconds
     - memory_usage
   * - predict
     - 1024
     - False
     - jax
     - 0.0066
     - 
   * - predict
     - 1024
     - False
     - pytorch
     - 0.0013
     - 
   * - predict
     - 1024
     - False
     - pytorch
     - 0.0022
     - 
   * - predict
     - 1024
     - False
     - tensorflow
     - 0.0072
     - 
   * - predict
     - 1024
     - False
     - tensorflow
     - 0.0136
     - 
   * - predict
     - 1024
     - True
     - jax
     - 0.3759
     - 
   * - predict
     - 1024
     - True
     - pytorch
     - 0.0033
     - 
   * - predict
     - 1024
     - True
     - pytorch
     - 0.0038
     - 
   * - predict
     - 1024
     - True
     - tensorflow
     - 0.0112
     - 
   * - predict
     - 1024
     - True
     - tensorflow
     - 0.0675
     - 
   * - predict
     - 2048
     - False
     - jax
     - 0.0122
     - 
   * - predict
     - 2048
     - False
     - pytorch
     - 0.0024
     - 
   * - predict
     - 2048
     - False
     - pytorch
     - 0.0043
     - 
   * - predict
     - 2048
     - False
     - tensorflow
     - 0.0178
     - 
   * - predict
     - 2048
     - False
     - tensorflow
     - 0.0188
     - 
   * - predict
     - 4096
     - False
     - jax
     - 0.0217
     - 
   * - predict
     - 4096
     - False
     - pytorch
     - 0.0033
     - 
   * - predict
     - 4096
     - False
     - pytorch
     - 0.0066
     - 
   * - predict
     - 4096
     - False
     - tensorflow
     - 0.0201
     - 
   * - predict
     - 4096
     - False
     - tensorflow
     - 0.0291
     - 
   * - predict
     - 8192
     - False
     - jax
     - 0.0402
     - 
   * - predict
     - 8192
     - False
     - pytorch
     - 0.0100
     - 
   * - predict
     - 8192
     - False
     - pytorch
     - 0.0123
     - 
   * - predict
     - 8192
     - False
     - tensorflow
     - 0.0385
     - 
   * - predict
     - 8192
     - False
     - tensorflow
     - 0.0472
     - 
   * - predict
     - 16384
     - False
     - jax
     - 0.0817
     - 
   * - predict
     - 16384
     - False
     - pytorch
     - 0.0140
     - 
   * - predict
     - 16384
     - False
     - pytorch
     - 0.0246
     - 
   * - predict
     - 16384
     - False
     - tensorflow
     - 0.0694
     - 
   * - predict
     - 16384
     - False
     - tensorflow
     - 0.0870
     - 
   * - predict
     - 32768
     - False
     - jax
     - 0.1843
     - 
   * - predict
     - 32768
     - False
     - pytorch
     - 0.0299
     - 
   * - predict
     - 32768
     - False
     - pytorch
     - 0.0458
     - 
   * - predict
     - 32768
     - False
     - tensorflow
     - 0.1383
     - 
   * - predict
     - 32768
     - False
     - tensorflow
     - 0.1586
     - 
   * - predict
     - 65536
     - False
     - jax
     - 0.3178
     - 
   * - predict
     - 65536
     - False
     - pytorch
     - 0.0420
     - 
   * - predict
     - 65536
     - False
     - pytorch
     - 0.0955
     - 
   * - predict
     - 65536
     - False
     - tensorflow
     - 0.2869
     - 
   * - predict
     - 65536
     - False
     - tensorflow
     - 0.3386
     - 
   * - predict
     - 131072
     - False
     - jax
     - 0.6357
     - 
   * - predict
     - 131072
     - False
     - pytorch
     - 0.1166
     - 
   * - predict
     - 131072
     - False
     - pytorch
     - 0.2265
     - 
   * - predict
     - 131072
     - False
     - tensorflow
     - 0.5704
     - 
   * - predict
     - 131072
     - False
     - tensorflow
     - 0.6503
     - 
   * - sample
     - 1024
     - False
     - jax
     - 0.7469
     - 
   * - sample
     - 1024
     - False
     - pytorch
     - 0.0064
     - 
   * - sample
     - 1024
     - False
     - pytorch
     - 0.0553
     - 
   * - sample
     - 1024
     - False
     - tensorflow
     - 0.0280
     - 
   * - sample
     - 1024
     - False
     - tensorflow
     - 0.0727
     - 
   * - sample
     - 1024
     - True
     - jax
     - 3.1792
     - 
   * - sample
     - 1024
     - True
     - pytorch
     - 0.0111
     - 
   * - sample
     - 1024
     - True
     - pytorch
     - 0.0766
     - 
   * - sample
     - 1024
     - True
     - tensorflow
     - 0.2347
     - 
   * - sample
     - 1024
     - True
     - tensorflow
     - 0.6430
     - 
   * - sample
     - 2048
     - False
     - jax
     - 1.5119
     - 
   * - sample
     - 2048
     - False
     - pytorch
     - 0.0142
     - 
   * - sample
     - 2048
     - False
     - pytorch
     - 0.1275
     - 
   * - sample
     - 2048
     - False
     - tensorflow
     - 0.0921
     - 
   * - sample
     - 2048
     - False
     - tensorflow
     - 0.1411
     - 
   * - sample
     - 4096
     - False
     - jax
     - 2.9452
     - 
   * - sample
     - 4096
     - False
     - pytorch
     - 0.0287
     - 
   * - sample
     - 4096
     - False
     - pytorch
     - 0.2504
     - 
   * - sample
     - 4096
     - False
     - tensorflow
     - 0.1124
     - 
   * - sample
     - 4096
     - False
     - tensorflow
     - 0.2623
     - 
   * - sample
     - 8192
     - False
     - jax
     - 7.0083
     - 
   * - sample
     - 8192
     - False
     - pytorch
     - 0.0605
     - 
   * - sample
     - 8192
     - False
     - pytorch
     - 0.5467
     - 
   * - sample
     - 8192
     - False
     - tensorflow
     - 0.2250
     - 
   * - sample
     - 8192
     - False
     - tensorflow
     - 0.5120
     - 
   * - sample
     - 16384
     - False
     - jax
     - 12.9754
     - 
   * - sample
     - 16384
     - False
     - pytorch
     - 0.1253
     - 
   * - sample
     - 16384
     - False
     - pytorch
     - 1.0257
     - 
   * - sample
     - 16384
     - False
     - tensorflow
     - 0.4454
     - 
   * - sample
     - 16384
     - False
     - tensorflow
     - 1.0652
     - 
   * - sample
     - 32768
     - False
     - jax
     - 26.1943
     - 
   * - sample
     - 32768
     - False
     - pytorch
     - 0.2521
     - 
   * - sample
     - 32768
     - False
     - pytorch
     - 3.0564
     - 
   * - sample
     - 32768
     - False
     - tensorflow
     - 0.8988
     - 
   * - sample
     - 32768
     - False
     - tensorflow
     - 2.7354
     - 
   * - sample
     - 65536
     - False
     - jax
     - 51.3290
     - 
   * - sample
     - 65536
     - False
     - pytorch
     - 0.5143
     - 
   * - sample
     - 65536
     - False
     - pytorch
     - 4.1766
     - 
   * - sample
     - 65536
     - False
     - tensorflow
     - 1.7781
     - 
   * - sample
     - 65536
     - False
     - tensorflow
     - 4.2621
     - 
   * - sample
     - 131072
     - False
     - jax
     - 101.0821
     - 
   * - sample
     - 131072
     - False
     - pytorch
     - 1.1038
     - 
   * - sample
     - 131072
     - False
     - pytorch
     - 9.1291
     - 
   * - sample
     - 131072
     - False
     - tensorflow
     - 3.5619
     - 
   * - sample
     - 131072
     - False
     - tensorflow
     - 11.2442
     - 
   * - train
     - 1024
     - False
     - jax
     - 6.3979
     - 1981649.0000
   * - train
     - 1024
     - False
     - pytorch
     - 0.8247
     - 54920.0000
   * - train
     - 1024
     - False
     - pytorch
     - 1.7825
     - 178457.0000
   * - train
     - 1024
     - False
     - tensorflow
     - 11.4120
     - 4262734.0000
   * - train
     - 1024
     - False
     - tensorflow
     - 15.2351
     - 4261753.0000
   * - train
     - 1024
     - True
     - jax
     - 233.3745
     - 212160901.0000
   * - train
     - 1024
     - True
     - pytorch
     - 14.2693
     - 67472934.0000
   * - train
     - 1024
     - True
     - pytorch
     - 38.9175
     - 78652464.0000
   * - train
     - 1024
     - True
     - tensorflow
     - 42.3649
     - 48039817.0000
   * - train
     - 1024
     - True
     - tensorflow
     - 43.4427
     - 48041726.0000
   * - train
     - 2048
     - False
     - jax
     - 5.4820
     - 1590916.0000
   * - train
     - 2048
     - False
     - pytorch
     - 1.0945
     - 58259.0000
   * - train
     - 2048
     - False
     - pytorch
     - 2.3677
     - 187586.0000
   * - train
     - 2048
     - False
     - tensorflow
     - 4.9678
     - 3045711.0000
   * - train
     - 2048
     - False
     - tensorflow
     - 6.3980
     - 3043035.0000
   * - train
     - 4096
     - False
     - jax
     - 8.0567
     - 1588851.0000
   * - train
     - 4096
     - False
     - pytorch
     - 1.8856
     - 67774.0000
   * - train
     - 4096
     - False
     - pytorch
     - 3.4129
     - 227195.0000
   * - train
     - 4096
     - False
     - tensorflow
     - 5.5753
     - 3051972.0000
   * - train
     - 4096
     - False
     - tensorflow
     - 6.4939
     - 3053964.0000
   * - train
     - 8192
     - False
     - jax
     - 11.0620
     - 1588058.0000
   * - train
     - 8192
     - False
     - pytorch
     - 4.5761
     - 85839.0000
   * - train
     - 8192
     - False
     - pytorch
     - 5.2089
     - 274424.0000
   * - train
     - 8192
     - False
     - tensorflow
     - 8.2938
     - 3073601.0000
   * - train
     - 8192
     - False
     - tensorflow
     - 9.3237
     - 3076821.0000
   * - train
     - 16384
     - False
     - jax
     - 16.3205
     - 1589611.0000
   * - train
     - 16384
     - False
     - pytorch
     - 6.6453
     - 123022.0000
   * - train
     - 16384
     - False
     - pytorch
     - 10.1101
     - 295999.0000
   * - train
     - 16384
     - False
     - tensorflow
     - 11.0820
     - 3132560.0000
   * - train
     - 16384
     - False
     - tensorflow
     - 14.3108
     - 3133822.0000
   * - train
     - 32768
     - False
     - jax
     - 28.7589
     - 1588602.0000
   * - train
     - 32768
     - False
     - pytorch
     - 14.9850
     - 139373.0000
   * - train
     - 32768
     - False
     - pytorch
     - 19.5333
     - 308842.0000
   * - train
     - 32768
     - False
     - tensorflow
     - 16.0620
     - 3158316.0000
   * - train
     - 32768
     - False
     - tensorflow
     - 25.2832
     - 3163477.0000
   * - train
     - 65536
     - False
     - jax
     - 53.7458
     - 1592431.0000
   * - train
     - 65536
     - False
     - pytorch
     - 27.2720
     - 139744.0000
   * - train
     - 65536
     - False
     - pytorch
     - 36.7689
     - 299354.0000
   * - train
     - 65536
     - False
     - tensorflow
     - 28.1539
     - 3167298.0000
   * - train
     - 65536
     - False
     - tensorflow
     - 47.7508
     - 3173626.0000
   * - train
     - 131072
     - False
     - jax
     - 102.4001
     - 1590250.0000
   * - train
     - 131072
     - False
     - pytorch
     - 57.1212
     - 137812.0000
   * - train
     - 131072
     - False
     - pytorch
     - 74.0363
     - 297794.0000
   * - train
     - 131072
     - False
     - tensorflow
     - 54.4925
     - 3159073.0000
   * - train
     - 131072
     - False
     - tensorflow
     - 91.8901
     - 3163690.0000
