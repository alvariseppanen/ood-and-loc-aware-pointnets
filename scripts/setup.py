#!/usr/bin/env python3

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

include_dirs=[np.get_include()]

setup(ext_modules = cythonize('custom_functions.pyx',compiler_directives={'language_level' : "3"}), include_dirs=[np.get_include()])
