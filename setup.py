import setuptools
from setuptools import setup

setup(name='graph_scout',
      version='0.7.5',
      description='A graph-based multi-agent environment scenario for scout mission simulations',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'scipy', 'networkx', 'gym'],
      python_requires='>=3.7',
      )
