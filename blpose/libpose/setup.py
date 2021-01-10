import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

source_file = os.path.join(os.path.dirname(__file__), "libpaf_cpu.pyx")

setup(
    name="libpaf",
    ext_modules=cythonize(source_file),
    include_dirs=[np.get_include()],
    py_modules="libpaf_cpu",
)
