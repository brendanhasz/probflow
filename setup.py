from setuptools import find_packages, setup

setup(name='probflow',
      version='0.1',
      description='A Keras-like interface for building variational Bayesian models with TensorFlow Probability',
      url='https://github.com/brendanhasz/probflow',
      author='Brendan Hasz',
      author_email='winsto99@gmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)