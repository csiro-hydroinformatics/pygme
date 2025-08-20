#!/usr/bin/env python

import os
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

# Define Cython C extensions
if os.getenv("PYGME_NO_BUILD") == "1":
    # Not extension
    extensions = []
else:
    extensions = [
        Extension(
            name="c_pygme_models_utils",
            sources=[
                "src/pygme/models/c_pygme_models_utils.pyx",
                "src/pygme/models/c_utils.c",
                "src/pygme/models/c_uh.c",
            ],
            include_dirs=[numpy.get_include()]),

        Extension(
            name="c_pygme_models_hydromodels",
            sources=[
                "src/pygme/models/c_pygme_models_hydromodels.pyx",
                "src/pygme/models/c_utils.c",
                "src/pygme/models/c_uh.c",
                "src/pygme/models/c_lagroute.c",
                "src/pygme/models/c_gr2m.c",
                "src/pygme/models/c_gr4j.c",
                "src/pygme/models/c_gr6j.c",
                "src/pygme/models/c_sac15.c",
                "src/pygme/models/c_wapaba.c",
                "src/pygme/models/c_ihacres.c"
            ],
            include_dirs=[numpy.get_include()]),
    ]

setup(
    name = "pygme",
    ext_modules = cythonize(extensions,
                            compiler_directives={"language_level": 3,
                                                 "profile": False})
)


