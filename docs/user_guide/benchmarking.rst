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


Comparing dimensionality
-------------------------

The plots below show runtime as a function of the number of datapoints, with a separate line for each number of dimensions.  Separate plots are shown for each backend and operation.  Only non-eager (compiled) training runs are included.

.. tabs::

    .. group-tab:: Train

        .. image:: ../img/benchmarking/dim_comparison_jax_train.png
           :width: 70 %
           :align: center

        .. image:: ../img/benchmarking/dim_comparison_pytorch_train.png
           :width: 70 %
           :align: center

        .. image:: ../img/benchmarking/dim_comparison_tensorflow_train.png
           :width: 70 %
           :align: center

    .. group-tab:: Predict

        .. image:: ../img/benchmarking/dim_comparison_jax_predict.png
           :width: 70 %
           :align: center

        .. image:: ../img/benchmarking/dim_comparison_pytorch_predict.png
           :width: 70 %
           :align: center

        .. image:: ../img/benchmarking/dim_comparison_tensorflow_predict.png
           :width: 70 %
           :align: center

    .. group-tab:: Sample

        .. image:: ../img/benchmarking/dim_comparison_jax_sample.png
           :width: 70 %
           :align: center

        .. image:: ../img/benchmarking/dim_comparison_pytorch_sample.png
           :width: 70 %
           :align: center

        .. image:: ../img/benchmarking/dim_comparison_tensorflow_sample.png
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
     - 0.0021
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
     - 0.0050
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
     - 0.0022
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
     - 0.0020
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
     - 0.0018
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
     - 0.0015
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
     - 0.0023
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
     - 0.0017
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
     - 0.0124
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
     - 0.0114
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
     - 0.0126
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
     - 0.0163
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
     - 0.1112
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
     - 0.0916
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
     - 0.0931
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
     - 0.0910
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
     - 0.0226
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
     - 0.1061
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
     - 0.0220
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
     - 0.0254
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
     - 0.0230
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
     - 0.0240
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
     - 0.0247
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
     - 0.0227
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
     - 0.1768
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
     - 0.1701
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
     - 0.1644
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
     - 0.1782
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
     - 1.5435
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
     - 1.3799
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
     - 1.4873
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
     - 1.5398
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
     - 1.7707
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
     - 6.5328
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
     - 0.7478
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
     - 3.0784
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
     - 0.6542
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
     - 2.3830
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
     - 0.6602
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
     - 2.3786
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
     - 1.0981
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
     - 1.1159
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
     - 1.1356
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
     - 1.2922
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
     - 5.1239
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
     - 5.7988
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
     - 5.3632
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
     - 5.8745
