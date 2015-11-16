#!/usr/bin/env python

import os
import numpy

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Distutils import build_ext


def read(fname):
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)).read()

# C extensions
ext_utils = Extension(name='c_useme_models_utils', 
    sources=[
        'useme/models/c_useme_models_utils.pyx', 
        'useme/models/c_utils.c',
        'useme/models/c_uh.c'
    ],
    include_dirs=[numpy.get_include()])

ext_gr4j = Extension(name='c_useme_models_gr4j', 
    sources=[
        'useme/models/c_useme_models_gr4j.pyx', 
        'useme/models/c_utils.c',
        'useme/models/c_uh.c',
        'useme/models/c_gr4j.c'
    ],
    include_dirs=[numpy.get_include()])

ext_gr2m = Extension(name='c_useme_models_gr2m', 
    sources=[
        'useme/models/c_useme_models_gr2m.pyx', 
        'useme/models/c_utils.c',
        'useme/models/c_gr2m.c'
    ],
    include_dirs=[numpy.get_include()])

ext_lagroute = Extension(name='c_useme_models_lagroute', 
    sources=[
        'useme/models/c_useme_models_lagroute.pyx', 
        'useme/models/c_utils.c',
        'useme/models/c_uh.c',
        'useme/models/c_lagroute.c'
    ],
    include_dirs=[numpy.get_include()])


# Package config
config = {
    'name': 'useme',
    'version': '0.1',
    'description': 'Design, run and calibrate models used in environmental sciences',
    'long_description': read('README.rst'),
    'author': 'Julien Lerat',
    'author_email': 'julien.lerat@gmail.com',
    'license': 'MIT',
    'url': 'https://bitbucket.org/jlerat/useme',
    'download_url': 'https://bitbucket.org/jlerat/useme/downloads',
    'install_requires': [
        'cython', \
        'numpy >= 1.8.0' \
    ],
    'cmdclass':{'build_ext':build_ext},
    'ext_modules':[
        ext_utils,
        ext_gr4j,
        ext_gr2m,
        ext_lagroute
    ],
    'test_suite':'nose.collector',
    'tests_require':['nose'],
    'classifiers':[
        'Operating System :: OS Independent', 
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: MIT License'
    ],
 
}

setup(**config)


