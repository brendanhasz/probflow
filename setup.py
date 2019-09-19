from setuptools import find_packages, setup

setup(name='probflow',
    version='2.0.0',
    description='A Python package for building Bayesian models with TensorFlow or PyTorch',
    url='https://github.com/brendanhasz/probflow',
    author='Brendan Hasz',
    author_email='winsto99@gmail.com',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
    extras_require={
        'tests': [
            'gast==0.2.2', #TODO: workaround for a TF req
            'tensorflow == 2.0.0-rc1',
            'tfp-nightly',
            'torch >= 1.2.0',
            'pytest >= 5.1.2',
            'pytest-cov >= 2.7.1',
            'pylint >= 2.3.1',
            ],
        'docs': [
            'tensorflow == 2.0.0-rc1',
            'tfp-nightly',
            'sphinx-tabs'
            ]
    }
)
