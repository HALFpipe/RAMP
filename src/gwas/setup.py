# -*- coding: utf-8 -*-
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Adapted from https://github.com/pandas-dev/pandas/blob/main/setup.py
debugging_symbols_requested = "--with-debugging-symbols" in sys.argv
if debugging_symbols_requested:
    sys.argv.remove("--with-debugging-symbols")

extra_compile_args: list[str] = ["-O3", "-std=c++20"]
extra_link_args: list[str] = ["-g3"]
if debugging_symbols_requested:
    extra_compile_args.append("-Wall")
    extra_compile_args.append("-Werror")
    extra_compile_args.append("-UNDEBUG")
    extra_compile_args.append("-O0")
    extra_compile_args.append("-march=native")


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
                libraries=["mkl_rt"],
            ),
            Extension(
                "gwas.mem._os",
                [
                    "src/gwas/mem/_os.pyx",
                ],
                define_macros=[
                    ("_GNU_SOURCE", None),
                    ("NPY_NO_DEPRECATED_API", None),
                    ("NDEBUG", None),
                ],
                include_dirs=[],
                libraries=[],
            ),
            Extension(
                "gwas.compression.arr._get",
                [
                    "src/gwas/compression/arr/_get.pyx",
                ],
                define_macros=[
                    ("NPY_NO_DEPRECATED_API", None),
                    ("NDEBUG", None),
                ],
                include_dirs=[np.get_include()],
                libraries=["blosc2"],
            ),
            Extension(
                "gwas.vcf._htslib",
                [
                    "src/gwas/vcf/_htslib.pyx",
                ],
                define_macros=[
                    ("NDEBUG", None),
                ],
                include_dirs=[np.get_include()],
                libraries=["hts"],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            ),
        ]
    )
    + [
        Extension(
            "gwas.vcf._read",
            ["src/gwas/vcf/_read.cpp"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    # rust_extensions=[
    #     RustExtension("gwas._rust", path="src/rust/Cargo.toml", binding=Binding.PyO3)
    # ],
    zip_safe=False,
)
