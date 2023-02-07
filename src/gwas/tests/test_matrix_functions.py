# -*- coding: utf-8 -*-
import numpy as np

from gwas._matrix_functions import dimatcopy


def test_dimatcopy_square():
    a = np.random.rand(8, 8)

    b = a.copy()
    dimatcopy(b)

    assert np.allclose(a.transpose(), b)


def test_dimatcopy_rect():
    a = np.random.rand(5, 8)

    b = a.copy()
    dimatcopy(b)

    assert np.allclose(
        a,
        np.ndarray(b.shape, b.dtype, buffer=b.data, order="F"),
    )
