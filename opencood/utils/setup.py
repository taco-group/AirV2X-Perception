from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name="box overlaps",
    ext_modules=cythonize("opencood/utils/box_overlaps.pyx"),
    include_dirs=[numpy.get_include()],
)
