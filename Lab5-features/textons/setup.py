from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs=[np.get_include()],
    ext_modules=cythonize([
        Extension('textons.kmc2', sources=[
                  'kmc2.c'], extra_compile_args=['-O3']),
        Extension("textons.lib_textons", ["lib_textons.pyx"])
    ]),
)
