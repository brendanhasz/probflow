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
     - n_dimensions
     - eager
     - backend
     - runtime_seconds
   * - predict
     - 1024
     - 100
     - False
     - jax
     - 0.0058
   * - predict
     - 1024
     - 100
     - False
     - pytorch
     - 0.0010
   * - predict
     - 1024
     - 100
     - False
     - pytorch
     - 0.0019
   * - predict
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0051
   * - predict
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0067
   * - predict
     - 1024
     - 100
     - True
     - jax
     - 0.2730
   * - predict
     - 1024
     - 100
     - True
     - pytorch
     - 0.0014
   * - predict
     - 1024
     - 100
     - True
     - pytorch
     - 0.0017
   * - predict
     - 1024
     - 100
     - True
     - tensorflow
     - 0.0141
   * - predict
     - 1024
     - 100
     - True
     - tensorflow
     - 0.0634
   * - predict
     - 2048
     - 100
     - False
     - jax
     - 0.0124
   * - predict
     - 2048
     - 100
     - False
     - pytorch
     - 0.0021
   * - predict
     - 2048
     - 100
     - False
     - pytorch
     - 0.0074
   * - predict
     - 2048
     - 100
     - False
     - tensorflow
     - 0.0098
   * - predict
     - 2048
     - 100
     - False
     - tensorflow
     - 0.0243
   * - predict
     - 4096
     - 100
     - False
     - jax
     - 0.0252
   * - predict
     - 4096
     - 100
     - False
     - pytorch
     - 0.0053
   * - predict
     - 4096
     - 100
     - False
     - pytorch
     - 0.0087
   * - predict
     - 4096
     - 100
     - False
     - tensorflow
     - 0.0180
   * - predict
     - 4096
     - 100
     - False
     - tensorflow
     - 0.0281
   * - predict
     - 8192
     - 100
     - False
     - jax
     - 0.0827
   * - predict
     - 8192
     - 100
     - False
     - pytorch
     - 0.0107
   * - predict
     - 8192
     - 100
     - False
     - pytorch
     - 0.0147
   * - predict
     - 8192
     - 100
     - False
     - tensorflow
     - 0.0449
   * - predict
     - 8192
     - 100
     - False
     - tensorflow
     - 0.0595
   * - predict
     - 16384
     - 100
     - False
     - jax
     - 0.1059
   * - predict
     - 16384
     - 100
     - False
     - pytorch
     - 0.0175
   * - predict
     - 16384
     - 100
     - False
     - pytorch
     - 0.0250
   * - predict
     - 16384
     - 100
     - False
     - tensorflow
     - 0.0695
   * - predict
     - 16384
     - 100
     - False
     - tensorflow
     - 0.0907
   * - predict
     - 32768
     - 100
     - False
     - jax
     - 0.2366
   * - predict
     - 32768
     - 100
     - False
     - pytorch
     - 0.0238
   * - predict
     - 32768
     - 100
     - False
     - pytorch
     - 0.0482
   * - predict
     - 32768
     - 100
     - False
     - tensorflow
     - 0.1370
   * - predict
     - 32768
     - 100
     - False
     - tensorflow
     - 0.2739
   * - predict
     - 65536
     - 100
     - False
     - jax
     - 0.4039
   * - predict
     - 65536
     - 100
     - False
     - pytorch
     - 0.0577
   * - predict
     - 65536
     - 100
     - False
     - pytorch
     - 0.0888
   * - predict
     - 65536
     - 100
     - False
     - tensorflow
     - 0.2828
   * - predict
     - 65536
     - 100
     - False
     - tensorflow
     - 0.4201
   * - predict
     - 131072
     - 100
     - False
     - jax
     - 0.7409
   * - predict
     - 131072
     - 100
     - False
     - pytorch
     - 0.1401
   * - predict
     - 131072
     - 100
     - False
     - pytorch
     - 0.1794
   * - predict
     - 131072
     - 100
     - False
     - tensorflow
     - 0.5687
   * - predict
     - 131072
     - 100
     - False
     - tensorflow
     - 0.8265
   * - sample
     - 1024
     - 100
     - False
     - jax
     - 0.7742
   * - sample
     - 1024
     - 100
     - False
     - pytorch
     - 0.0062
   * - sample
     - 1024
     - 100
     - False
     - pytorch
     - 0.0827
   * - sample
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0293
   * - sample
     - 1024
     - 100
     - False
     - tensorflow
     - 0.0986
   * - sample
     - 1024
     - 100
     - True
     - jax
     - 3.0918
   * - sample
     - 1024
     - 100
     - True
     - pytorch
     - 0.0106
   * - sample
     - 1024
     - 100
     - True
     - pytorch
     - 0.0822
   * - sample
     - 1024
     - 100
     - True
     - tensorflow
     - 0.3750
   * - sample
     - 1024
     - 100
     - True
     - tensorflow
     - 0.6235
   * - sample
     - 2048
     - 100
     - False
     - jax
     - 1.5520
   * - sample
     - 2048
     - 100
     - False
     - pytorch
     - 0.0299
   * - sample
     - 2048
     - 100
     - False
     - pytorch
     - 0.1685
   * - sample
     - 2048
     - 100
     - False
     - tensorflow
     - 0.0595
   * - sample
     - 2048
     - 100
     - False
     - tensorflow
     - 0.2429
   * - sample
     - 4096
     - 100
     - False
     - jax
     - 3.9561
   * - sample
     - 4096
     - 100
     - False
     - pytorch
     - 0.0537
   * - sample
     - 4096
     - 100
     - False
     - pytorch
     - 0.5410
   * - sample
     - 4096
     - 100
     - False
     - tensorflow
     - 0.1088
   * - sample
     - 4096
     - 100
     - False
     - tensorflow
     - 0.2772
   * - sample
     - 8192
     - 100
     - False
     - jax
     - 7.9530
   * - sample
     - 8192
     - 100
     - False
     - pytorch
     - 0.0743
   * - sample
     - 8192
     - 100
     - False
     - pytorch
     - 1.0907
   * - sample
     - 8192
     - 100
     - False
     - tensorflow
     - 0.3253
   * - sample
     - 8192
     - 100
     - False
     - tensorflow
     - 0.6048
   * - sample
     - 16384
     - 100
     - False
     - jax
     - 15.9989
   * - sample
     - 16384
     - 100
     - False
     - pytorch
     - 0.1209
   * - sample
     - 16384
     - 100
     - False
     - pytorch
     - 1.2863
   * - sample
     - 16384
     - 100
     - False
     - tensorflow
     - 0.6584
   * - sample
     - 16384
     - 100
     - False
     - tensorflow
     - 1.2377
   * - sample
     - 32768
     - 100
     - False
     - jax
     - 39.8401
   * - sample
     - 32768
     - 100
     - False
     - pytorch
     - 0.2470
   * - sample
     - 32768
     - 100
     - False
     - pytorch
     - 2.6229
   * - sample
     - 32768
     - 100
     - False
     - tensorflow
     - 0.9259
   * - sample
     - 32768
     - 100
     - False
     - tensorflow
     - 3.3888
   * - sample
     - 65536
     - 100
     - False
     - jax
     - 65.7225
   * - sample
     - 65536
     - 100
     - False
     - pytorch
     - 0.5195
   * - sample
     - 65536
     - 100
     - False
     - pytorch
     - 5.9163
   * - sample
     - 65536
     - 100
     - False
     - tensorflow
     - 1.8565
   * - sample
     - 65536
     - 100
     - False
     - tensorflow
     - 7.4134
   * - sample
     - 131072
     - 100
     - False
     - jax
     - 121.1876
   * - sample
     - 131072
     - 100
     - False
     - pytorch
     - 0.9666
   * - sample
     - 131072
     - 100
     - False
     - pytorch
     - 9.5536
   * - sample
     - 131072
     - 100
     - False
     - tensorflow
     - 4.4857
   * - sample
     - 131072
     - 100
     - False
     - tensorflow
     - 11.1786
   * - train
     - 1024
     - 100
     - False
     - jax
     - 2.9958
   * - train
     - 1024
     - 100
     - False
     - pytorch
     - 0.6191
   * - train
     - 1024
     - 100
     - False
     - pytorch
     - 1.3979
   * - train
     - 1024
     - 100
     - False
     - tensorflow
     - 2.9685
   * - train
     - 1024
     - 100
     - False
     - tensorflow
     - 3.8661
   * - train
     - 1024
     - 100
     - True
     - jax
     - 84.3623
   * - train
     - 1024
     - 100
     - True
     - pytorch
     - 1.9310
   * - train
     - 1024
     - 100
     - True
     - pytorch
     - 2.1963
   * - train
     - 1024
     - 100
     - True
     - tensorflow
     - 14.6899
   * - train
     - 1024
     - 100
     - True
     - tensorflow
     - 23.8260
   * - train
     - 2048
     - 100
     - False
     - jax
     - 5.3753
   * - train
     - 2048
     - 100
     - False
     - pytorch
     - 0.8301
   * - train
     - 2048
     - 100
     - False
     - pytorch
     - 2.6966
   * - train
     - 2048
     - 100
     - False
     - tensorflow
     - 3.3068
   * - train
     - 2048
     - 100
     - False
     - tensorflow
     - 3.6000
   * - train
     - 4096
     - 100
     - False
     - jax
     - 4.3393
   * - train
     - 4096
     - 100
     - False
     - pytorch
     - 1.6704
   * - train
     - 4096
     - 100
     - False
     - pytorch
     - 4.9118
   * - train
     - 4096
     - 100
     - False
     - tensorflow
     - 3.1260
   * - train
     - 4096
     - 100
     - False
     - tensorflow
     - 3.5091
   * - train
     - 8192
     - 100
     - False
     - jax
     - 7.6445
   * - train
     - 8192
     - 100
     - False
     - pytorch
     - 3.3542
   * - train
     - 8192
     - 100
     - False
     - pytorch
     - 5.1019
   * - train
     - 8192
     - 100
     - False
     - tensorflow
     - 3.6932
   * - train
     - 8192
     - 100
     - False
     - tensorflow
     - 6.2422
   * - train
     - 16384
     - 100
     - False
     - jax
     - 12.0983
   * - train
     - 16384
     - 100
     - False
     - pytorch
     - 5.5577
   * - train
     - 16384
     - 100
     - False
     - pytorch
     - 6.9370
   * - train
     - 16384
     - 100
     - False
     - tensorflow
     - 7.3068
   * - train
     - 16384
     - 100
     - False
     - tensorflow
     - 10.4082
   * - train
     - 32768
     - 100
     - False
     - jax
     - 24.8041
   * - train
     - 32768
     - 100
     - False
     - pytorch
     - 9.7383
   * - train
     - 32768
     - 100
     - False
     - pytorch
     - 12.4647
   * - train
     - 32768
     - 100
     - False
     - tensorflow
     - 12.6115
   * - train
     - 32768
     - 100
     - False
     - tensorflow
     - 21.2656
   * - train
     - 65536
     - 100
     - False
     - jax
     - 41.9768
   * - train
     - 65536
     - 100
     - False
     - pytorch
     - 17.5923
   * - train
     - 65536
     - 100
     - False
     - pytorch
     - 22.6260
   * - train
     - 65536
     - 100
     - False
     - tensorflow
     - 26.1306
   * - train
     - 65536
     - 100
     - False
     - tensorflow
     - 42.6369
   * - train
     - 131072
     - 100
     - False
     - jax
     - 83.3080
   * - train
     - 131072
     - 100
     - False
     - pytorch
     - 36.0449
   * - train
     - 131072
     - 100
     - False
     - pytorch
     - 46.3188
   * - train
     - 131072
     - 100
     - False
     - tensorflow
     - 47.6014
   * - train
     - 131072
     - 100
     - False
     - tensorflow
     - 80.0065
