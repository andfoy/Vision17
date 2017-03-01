from setuptools import setup, Extension
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs=[np.get_include()],
    ext_modules=cythonize([
        Extension("lib_textons", ["lib_textons.pyx"])
    ]),
    install_requires=["numpy", "scipy", "scikit-learn", "nose", "kmc2"],
    dependency_links=[
        "https://github.com/obachem/kmc2/archive/master.zip#egg=kmc2"]

)
