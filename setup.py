#!/usr/bin/env python

import os
import numpy as np
from setuptools import setup, find_packages, Extension

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

lpnorm_extension = Extension('lpnorm._lpnorm',
        sources=['lpnorm/lpnorm.c'],
        extra_compile_args=['-O3', '-march=native', '-fopenmp', '-std=c99'],
        extra_link_args=['-fopenmp'])

smmprod_extension = Extension('smmprod._smmprod',
        sources=['smmprod/smmprod.c'],
        extra_compile_args=['-O3', '-march=native', '-fopenmp', '-std=c99'],
        extra_link_args=['-fopenmp'],
        libraries=['cblas'])

setup(
    name='grslra',
    version='0.1',
    description='Grassmannian Robust Structured (and Unstructured) Low-Rank Approximation',
    author='Clemens Hage',
    author_email='hage@tum.de',
    license='TUM',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'matplotlib'],
    include_dirs=[np.get_include()],
    ext_modules=[lpnorm_extension, smmprod_extension],
)
