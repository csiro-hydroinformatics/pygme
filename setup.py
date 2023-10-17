#!/usr/bin/env python

import os
import numpy
from pathlib import Path

from setuptools import setup, Extension, find_packages

from Cython.Distutils import build_ext

import versioneer

def read(fname):
    thisfolder = Path(__file__).resolve().parent
    with (thisfolder / fname).open() as fo:
        return fo.read()

# C extensions
if os.getenv('PYGME_NO_BUILD') == '1':
    # Not extension
    ext_modules = []
else:
    ext_utils=Extension(name='c_pygme_models_utils',
        sources=[
            'pygme/models/c_pygme_models_utils.pyx',
            'pygme/models/c_utils.c',
            'pygme/models/c_uh.c'
        ],
        extra_cflags=['-O3'],
        extra_compile_args=['-ffast-math'],
        include_dirs=[numpy.get_include()])

    ext_hydromodels=Extension(name='c_pygme_models_hydromodels',
        sources=[
            'pygme/models/c_pygme_models_hydromodels.pyx',
            'pygme/models/c_utils.c',
            'pygme/models/c_uh.c',
            'pygme/models/c_lagroute.c',
            'pygme/models/c_gr2m.c',
            'pygme/models/c_gr4j.c',
            'pygme/models/c_gr6j.c',
            'pygme/models/c_sac15.c',
            'pygme/models/c_wapaba.c',
            'pygme/models/c_ihacres.c'
        ],
        extra_cflags=['-O3'],
        extra_compile_args=['-ffast-math'],
        include_dirs=[numpy.get_include()])

    ext_modules = [
        ext_utils,
        ext_hydromodels
    ]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

# Package config
setup(
    name='pygme',
    author= 'Julien Lerat',
    author_email= 'julien.lerat@csiro.au',
    url= 'https://github.com/csiro-hydroinformatics/pygme',
    download_url= 'hhttps://github.com/csiro-hydroinformatics/pygme/tags',
    version=versioneer.get_version(),
    description= 'Design, run and calibrate models used in'+\
                            ' environmental sciences',
    long_description= read('README.rst'),
    packages=find_packages(),
    package_data={
        'pygme': [
            'tests/*.zip'
        ],
    },
    install_requires= [
        'cython',
        'hydrodiy',
        'numpy >= 1.8.0',
        'scipy (>=0.14.0)',
        'pandas >= 0.16'
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License'
    ]
)


