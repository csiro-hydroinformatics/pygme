#!/usr/bin/env python
import os

from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import versioneer


HBV_SRC = [
    "hbv/hbv_kernel.f90"
]
OPTS = '-O3 -ffast-math -funroll-loops'

setup(
    name='hbv',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    setup_requires=[
        "numpy>=1.8.1",
    ],
    tests_require=[
        "pandas>=0.14.1",
    ],
    ext_modules=[
        Extension(
            name='hbv_kernel',
            sources=HBV_SRC,
            extra_f90_compile_args=[OPTS]
        ),
    ],
    # Metadata
    description='Python wrapper for HBV.',
    author='J. Lerat'
)
