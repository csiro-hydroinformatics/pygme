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

ext_gr4j=Extension(name='c_pygme_models_gr4j',
    sources=[
        'pygme/models/c_pygme_models_gr4j.pyx',
        'pygme/models/c_utils.c',
        'pygme/models/c_uh.c',
        'pygme/models/c_gr4j.c'
    ],
    include_dirs=[numpy.get_include()])

ext_gr2m=Extension(name='c_pygme_models_gr2m',
    sources=[
        'pygme/models/c_pygme_models_gr2m.pyx',
        'pygme/models/c_utils.c',
        'pygme/models/c_gr2m.c'
    ],
    include_dirs=[numpy.get_include()])

ext_lagroute=Extension(name='c_pygme_models_lagroute',
    sources=[
        'pygme/models/c_pygme_models_lagroute.pyx',
        'pygme/models/c_utils.c',
        'pygme/models/c_uh.c',
        'pygme/models/c_lagroute.c'
    ],
    include_dirs=[numpy.get_include()])

ext_knn=Extension(name='c_pygme_models_knndaily',
    sources=[
        'pygme/models/c_pygme_models_knndaily.pyx',
        'pygme/models/c_utils.c',
        'pygme/models/c_knndaily.c',
    ],
    include_dirs=[numpy.get_include()])


ext_basics=Extension(name='c_pygme_models_basics',
    sources=[
        'pygme/models/c_pygme_models_basics.pyx',
        'pygme/models/c_utils.c',
        'pygme/models/c_basics.c',
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
        'cython', \
        'numpy >= 1.8.0' \
        'pandas >= 0.16' \
    ],
    cmdclass=cmdclass,
    ext_modules=[
        ext_utils,
        ext_gr4j,
        ext_gr2m,
        ext_lagroute,
        ext_knn,
        ext_basics
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: MIT License'
    ]
)


