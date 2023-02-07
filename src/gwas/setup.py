# -*- coding: utf-8 -*-
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np

setup(
    ext_modules=cythonize([
        Extension(
            "gwas._matrix_functions",
            ["src/gwas/_matrix_functions.pyx"],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", None),
                ("NDEBUG", None),
            ],
            include_dirs=[np.get_include()],
            libraries=[
                "mkl_rt",
            ],
        ),
    ]),
    use_scm_version=dict(version_scheme="no-guess-dev"),
    zip_safe=False,
)
