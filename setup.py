from setuptools import find_packages, setup

setup(name='probflow',
      version='2.0',
      description='A Python package for building Bayesian models with TensorFlow or PyTorch',
      url='https://github.com/brendanhasz/probflow',
      author='Brendan Hasz',
      author_email='winsto99@gmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)