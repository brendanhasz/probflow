.. _user_guide_benchmarking:

Benchmarking
============

.. include:: ../macros.hrst

ProbFlow's benchmarking suite fits a Bayesian linear regression model (:class:`.LinearRegression`) with each supported backend (TensorFlow, PyTorch, and JAX), for a range of dataset sizes and data dimensionality.  For each combination it measures the time to train the model, the time to generate predictions, and the time to draw samples from the model's predictive distribution.  Training is additionally benchmarked in both eager and non-eager (compiled) modes, for smaller datasets (for larger datasets non-eager/compiled mode is always used).

Training times using eager vs compiled
--------------------------------------

The plot below compares training runtime in eager vs non-eager (compiled) mode for each backend, using the smallest number of datapoints and the largest number of dimensions benchmarked.

.. image:: ../img/benchmarking/eager_vs_noneager.png
   :width: 70 %
   :align: center

Training times across backends
------------------------------

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


Full benchmarking results
-------------------------

The full set of benchmarking results:

.. list-table::
   :header-rows: 1

   * - operation
     - n_datapoints
     - n_dimensions
     - eager
     - backend
     - runtime_seconds
   * - predict
     - 1024
     - 1
     - False
     - jax
     - 0.0021
   * - predict
     - 1024
     - 1
     - False
     - pytorch
     - 0.0010
   * - predict
     - 1024
     - 1
     - False
     - tensorflow
     - 0.0033
   * - predict
     - 1024
     - 1
     - True
     - jax
     - 0.1071
   * - predict
     - 1024
     - 1
     - True
     - pytorch
     - 0.0003
   * - predict
     - 1024
     - 1
     - True
     - tensorflow
     - 0.0085
   * - predict
     - 1024
     - 2
     - False
     - jax
     - 0.0023
   * - predict
     - 1024
     - 2
     - False
     - pytorch
     - 0.0004
   * - predict
     - 1024
     - 2
     - False
     - tensorflow
     - 0.0031
   * - predict
     - 1024
     - 2
     - True
     - jax
     - 0.0449
   * - predict
     - 1024
     - 2
     - True
     - pytorch
     - 0.0005
   * - predict
     - 1024
     - 2
     - True
     - tensorflow
     - 0.0031
   * - predict
     - 1024
     - 10
     - False
     - jax
     - 0.0023
   * - predict
     - 1024
     - 10
     - False
     - pytorch
     - 0.0018
   * - predict
     - 1024
     - 10
     - False
     - tensorflow
     - 0.0034
   * - predict
     - 1024
     - 10
     - True
     - jax
     - 0.0467
   * - predict
     - 1024
     - 10
     - True
     - pytorch
     - 0.0012
   * - predict
     - 1024
     - 10
     - True
     - tensorflow
     - 0.0027
   * - predict
     - 1024
     - 100
     - False
     - jax
     - 0.0025
   * - predict
     - 1024
     - 100
     - False
     - pytorch
     - 0.0003
   * - predict
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0032
   * - predict
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0032
   * - predict
     - 1024
     - 100
     - True
     - jax
     - 0.0630
   * - predict
     - 1024
     - 100
     - True
     - pytorch
     - 0.0003
   * - predict
     - 1024
     - 100
     - True
     - tensorflow
     - 0.0028
   * - predict
     - 1024
     - 100
     - True
     - tensorflow
     - 0.0064
   * - predict
     - 2048
     - 100
     - False
     - tensorflow
     - 0.0055
   * - predict
     - 4096
     - 100
     - False
     - tensorflow
     - 0.0094
   * - predict
     - 8192
     - 1
     - False
     - jax
     - 0.0154
   * - predict
     - 8192
     - 1
     - False
     - pytorch
     - 0.0016
   * - predict
     - 8192
     - 1
     - False
     - tensorflow
     - 0.0197
   * - predict
     - 8192
     - 2
     - False
     - jax
     - 0.0162
   * - predict
     - 8192
     - 2
     - False
     - pytorch
     - 0.0016
   * - predict
     - 8192
     - 2
     - False
     - tensorflow
     - 0.0200
   * - predict
     - 8192
     - 10
     - False
     - jax
     - 0.0144
   * - predict
     - 8192
     - 10
     - False
     - pytorch
     - 0.0014
   * - predict
     - 8192
     - 10
     - False
     - tensorflow
     - 0.0198
   * - predict
     - 8192
     - 100
     - False
     - jax
     - 0.0145
   * - predict
     - 8192
     - 100
     - False
     - pytorch
     - 0.0019
   * - predict
     - 8192
     - 100
     - False
     - tensorflow
     - 0.0181
   * - predict
     - 8192
     - 100
     - False
     - tensorflow
     - 0.0211
   * - predict
     - 16384
     - 100
     - False
     - tensorflow
     - 0.0324
   * - predict
     - 32768
     - 100
     - False
     - tensorflow
     - 0.0631
   * - predict
     - 65536
     - 1
     - False
     - jax
     - 0.1205
   * - predict
     - 65536
     - 1
     - False
     - pytorch
     - 0.0152
   * - predict
     - 65536
     - 1
     - False
     - tensorflow
     - 0.1557
   * - predict
     - 65536
     - 2
     - False
     - jax
     - 0.1057
   * - predict
     - 65536
     - 2
     - False
     - pytorch
     - 0.0091
   * - predict
     - 65536
     - 2
     - False
     - tensorflow
     - 0.1582
   * - predict
     - 65536
     - 10
     - False
     - jax
     - 0.1236
   * - predict
     - 65536
     - 10
     - False
     - pytorch
     - 0.0122
   * - predict
     - 65536
     - 10
     - False
     - tensorflow
     - 0.1624
   * - predict
     - 65536
     - 100
     - False
     - jax
     - 0.1196
   * - predict
     - 65536
     - 100
     - False
     - pytorch
     - 0.0319
   * - predict
     - 65536
     - 100
     - False
     - tensorflow
     - 0.1269
   * - predict
     - 65536
     - 100
     - False
     - tensorflow
     - 0.1700
   * - predict
     - 131072
     - 100
     - False
     - tensorflow
     - 0.2502
   * - sample
     - 1024
     - 1
     - False
     - jax
     - 0.1959
   * - sample
     - 1024
     - 1
     - False
     - pytorch
     - 0.0283
   * - sample
     - 1024
     - 1
     - False
     - tensorflow
     - 0.0337
   * - sample
     - 1024
     - 1
     - True
     - jax
     - 0.9256
   * - sample
     - 1024
     - 1
     - True
     - pytorch
     - 0.0156
   * - sample
     - 1024
     - 1
     - True
     - tensorflow
     - 0.1471
   * - sample
     - 1024
     - 2
     - False
     - jax
     - 0.1777
   * - sample
     - 1024
     - 2
     - False
     - pytorch
     - 0.0160
   * - sample
     - 1024
     - 2
     - False
     - tensorflow
     - 0.0314
   * - sample
     - 1024
     - 2
     - True
     - jax
     - 0.3198
   * - sample
     - 1024
     - 2
     - True
     - pytorch
     - 0.0197
   * - sample
     - 1024
     - 2
     - True
     - tensorflow
     - 0.0361
   * - sample
     - 1024
     - 10
     - False
     - jax
     - 0.1891
   * - sample
     - 1024
     - 10
     - False
     - pytorch
     - 0.0192
   * - sample
     - 1024
     - 10
     - False
     - tensorflow
     - 0.0329
   * - sample
     - 1024
     - 10
     - True
     - jax
     - 0.3664
   * - sample
     - 1024
     - 10
     - True
     - pytorch
     - 0.0174
   * - sample
     - 1024
     - 10
     - True
     - tensorflow
     - 0.0372
   * - sample
     - 1024
     - 100
     - False
     - jax
     - 0.1875
   * - sample
     - 1024
     - 100
     - False
     - pytorch
     - 0.0268
   * - sample
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0331
   * - sample
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0344
   * - sample
     - 1024
     - 100
     - True
     - jax
     - 0.4127
   * - sample
     - 1024
     - 100
     - True
     - pytorch
     - 0.0331
   * - sample
     - 1024
     - 100
     - True
     - tensorflow
     - 0.0419
   * - sample
     - 1024
     - 100
     - True
     - tensorflow
     - 0.1216
   * - sample
     - 2048
     - 100
     - False
     - tensorflow
     - 0.0788
   * - sample
     - 4096
     - 100
     - False
     - tensorflow
     - 0.1381
   * - sample
     - 8192
     - 1
     - False
     - jax
     - 1.4423
   * - sample
     - 8192
     - 1
     - False
     - pytorch
     - 0.1377
   * - sample
     - 8192
     - 1
     - False
     - tensorflow
     - 0.2908
   * - sample
     - 8192
     - 2
     - False
     - jax
     - 1.4391
   * - sample
     - 8192
     - 2
     - False
     - pytorch
     - 0.0939
   * - sample
     - 8192
     - 2
     - False
     - tensorflow
     - 0.2687
   * - sample
     - 8192
     - 10
     - False
     - jax
     - 1.4363
   * - sample
     - 8192
     - 10
     - False
     - pytorch
     - 0.0934
   * - sample
     - 8192
     - 10
     - False
     - tensorflow
     - 0.2888
   * - sample
     - 8192
     - 100
     - False
     - jax
     - 1.4904
   * - sample
     - 8192
     - 100
     - False
     - pytorch
     - 0.2029
   * - sample
     - 8192
     - 100
     - False
     - tensorflow
     - 0.2596
   * - sample
     - 8192
     - 100
     - False
     - tensorflow
     - 0.3162
   * - sample
     - 16384
     - 100
     - False
     - tensorflow
     - 0.5327
   * - sample
     - 32768
     - 100
     - False
     - tensorflow
     - 1.0417
   * - sample
     - 65536
     - 1
     - False
     - jax
     - 12.0752
   * - sample
     - 65536
     - 1
     - False
     - pytorch
     - 1.0688
   * - sample
     - 65536
     - 1
     - False
     - tensorflow
     - 2.3177
   * - sample
     - 65536
     - 2
     - False
     - jax
     - 12.8104
   * - sample
     - 65536
     - 2
     - False
     - pytorch
     - 0.7253
   * - sample
     - 65536
     - 2
     - False
     - tensorflow
     - 2.5890
   * - sample
     - 65536
     - 10
     - False
     - jax
     - 12.8615
   * - sample
     - 65536
     - 10
     - False
     - pytorch
     - 0.9260
   * - sample
     - 65536
     - 10
     - False
     - tensorflow
     - 2.3435
   * - sample
     - 65536
     - 100
     - False
     - jax
     - 13.2683
   * - sample
     - 65536
     - 100
     - False
     - pytorch
     - 2.0836
   * - sample
     - 65536
     - 100
     - False
     - tensorflow
     - 2.1398
   * - sample
     - 65536
     - 100
     - False
     - tensorflow
     - 2.7709
   * - sample
     - 131072
     - 100
     - False
     - tensorflow
     - 4.7513
   * - train
     - 1024
     - 1
     - False
     - jax
     - 0.7825
   * - train
     - 1024
     - 1
     - False
     - pytorch
     - 0.4383
   * - train
     - 1024
     - 1
     - False
     - tensorflow
     - 2.6671
   * - train
     - 1024
     - 1
     - True
     - jax
     - 21.9519
   * - train
     - 1024
     - 1
     - True
     - pytorch
     - 4.6179
   * - train
     - 1024
     - 1
     - True
     - tensorflow
     - 5.6455
   * - train
     - 1024
     - 2
     - False
     - jax
     - 0.7420
   * - train
     - 1024
     - 2
     - False
     - pytorch
     - 0.3008
   * - train
     - 1024
     - 2
     - False
     - tensorflow
     - 1.1905
   * - train
     - 1024
     - 2
     - True
     - jax
     - 18.9255
   * - train
     - 1024
     - 2
     - True
     - pytorch
     - 0.5095
   * - train
     - 1024
     - 2
     - True
     - tensorflow
     - 4.4745
   * - train
     - 1024
     - 10
     - False
     - jax
     - 0.7395
   * - train
     - 1024
     - 10
     - False
     - pytorch
     - 0.3970
   * - train
     - 1024
     - 10
     - False
     - tensorflow
     - 1.2027
   * - train
     - 1024
     - 10
     - True
     - jax
     - 20.0666
   * - train
     - 1024
     - 10
     - True
     - pytorch
     - 0.2536
   * - train
     - 1024
     - 10
     - True
     - tensorflow
     - 4.3776
   * - train
     - 1024
     - 100
     - False
     - jax
     - 0.7638
   * - train
     - 1024
     - 100
     - False
     - pytorch
     - 0.9779
   * - train
     - 1024
     - 100
     - False
     - tensorflow
     - 1.1892
   * - train
     - 1024
     - 100
     - False
     - tensorflow
     - 2.6654
   * - train
     - 1024
     - 100
     - True
     - jax
     - 21.6070
   * - train
     - 1024
     - 100
     - True
     - pytorch
     - 0.2846
   * - train
     - 1024
     - 100
     - True
     - tensorflow
     - 4.4074
   * - train
     - 1024
     - 100
     - True
     - tensorflow
     - 5.6328
   * - train
     - 2048
     - 100
     - False
     - tensorflow
     - 1.4975
   * - train
     - 4096
     - 100
     - False
     - tensorflow
     - 1.3429
   * - train
     - 8192
     - 1
     - False
     - jax
     - 1.9150
   * - train
     - 8192
     - 1
     - False
     - pytorch
     - 0.8115
   * - train
     - 8192
     - 1
     - False
     - tensorflow
     - 2.0973
   * - train
     - 8192
     - 2
     - False
     - jax
     - 1.9208
   * - train
     - 8192
     - 2
     - False
     - pytorch
     - 0.8963
   * - train
     - 8192
     - 2
     - False
     - tensorflow
     - 1.9994
   * - train
     - 8192
     - 10
     - False
     - jax
     - 1.8534
   * - train
     - 8192
     - 10
     - False
     - pytorch
     - 0.7565
   * - train
     - 8192
     - 10
     - False
     - tensorflow
     - 2.1315
   * - train
     - 8192
     - 100
     - False
     - jax
     - 2.1196
   * - train
     - 8192
     - 100
     - False
     - pytorch
     - 1.0277
   * - train
     - 8192
     - 100
     - False
     - tensorflow
     - 1.8096
   * - train
     - 8192
     - 100
     - False
     - tensorflow
     - 2.6268
   * - train
     - 16384
     - 100
     - False
     - tensorflow
     - 2.6402
   * - train
     - 32768
     - 100
     - False
     - tensorflow
     - 4.3746
   * - train
     - 65536
     - 1
     - False
     - jax
     - 12.2566
   * - train
     - 65536
     - 1
     - False
     - pytorch
     - 5.9947
   * - train
     - 65536
     - 1
     - False
     - tensorflow
     - 8.4454
   * - train
     - 65536
     - 2
     - False
     - jax
     - 12.4524
   * - train
     - 65536
     - 2
     - False
     - pytorch
     - 6.3375
   * - train
     - 65536
     - 2
     - False
     - tensorflow
     - 9.6065
   * - train
     - 65536
     - 10
     - False
     - jax
     - 11.6966
   * - train
     - 65536
     - 10
     - False
     - pytorch
     - 9.5214
   * - train
     - 65536
     - 10
     - False
     - tensorflow
     - 10.1852
   * - train
     - 65536
     - 100
     - False
     - jax
     - 12.8164
   * - train
     - 65536
     - 100
     - False
     - pytorch
     - 9.3513
   * - train
     - 65536
     - 100
     - False
     - tensorflow
     - 7.9833
   * - train
     - 65536
     - 100
     - False
     - tensorflow
     - 10.2184
   * - train
     - 131072
     - 100
     - False
     - tensorflow
     - 14.8537
