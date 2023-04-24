# -*- coding: utf-8 -*-
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "gwas._matrix_functions",
                [
                    "src/gwas/_tri.c",
                    "src/gwas/_matrix_functions.pyx",
                ],
                define_macros=[
                    ("NPY_NO_DEPRECATED_API", None),
                    ("NDEBUG", None),
                ],
                include_dirs=[np.get_include()],
                libraries=[
                    "mkl_rt",
                ],
            ),
            Extension(
                "gwas._os",
                [
                    "src/gwas/_os.pyx",
                ],
                define_macros=[
                    ("NPY_NO_DEPRECATED_API", None),
                    ("NDEBUG", None),
                ],
                include_dirs=[],
                libraries=[],
            ),
        ]
    ),
    zip_safe=False,
)
