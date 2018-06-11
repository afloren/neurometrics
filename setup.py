#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(name = 'neurometrics',
          version = '0.1',
          packages = find_packages(),
          install_requires = [
              'h5py',
              'nilearn',
              'nipy',
              'nipype',
              'parse',
              'pymvpa2',
              'pyspark',
              'scikit-neuralnetwork',
              'sklearn'
          ])
