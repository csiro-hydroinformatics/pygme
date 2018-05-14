#!/usr/bin/env python

import os
import numpy

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Distutils import build_ext

import versioneer

def read(fname):
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)).read()

# C extensions
ext_utils=Extension(name='c_pygme_models_utils',
    sources=[
        'pygme/models/c_pygme_models_utils.pyx',
        'pygme/models/c_utils.c',
        'pygme/models/c_uh.c'
    ],
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
        'pygme/models/c_hbv.c'
    ],
    include_dirs=[numpy.get_include()])

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

# Package config
setup(
    name='pygme',
    version= versioneer.get_version(),
    description= 'Design, run and calibrate models used in environmental sciences',
    long_description= read('README.rst'),
    author= 'Julien Lerat',
    author_email= 'julien.lerat@gmail.com',
    license= 'MIT',
    url= 'https://bitbucket.org/jlerat/pygme',
    download_url= 'https://bitbucket.org/jlerat/pygme/downloads',
    install_requires= [
        'cython',
        'numpy >= 1.8.0',
        'scipy (>=0.14.0)',
        'pandas >= 0.16'
    ],
    cmdclass=cmdclass,
    ext_modules=[
        ext_utils,
        ext_hydromodels
    ],
    test_suite='nose.collector',
    tests_require=['nose', 'hydrodiy'],
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: BSD License'
    ]
)


