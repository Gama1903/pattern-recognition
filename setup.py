from setuptools import setup, find_packages

setup(
    name='pattern-recognition',
    version='0.1',
    author='Gama1903',
    author_email='gama1903@qq.com',
    description=
    'An pattern-recognition method package, only includes \'bayesian classifier\' currently',
    packages=find_packages(),
    install_requires=['os', 'torch', 'pandas', 'sklearn'])
