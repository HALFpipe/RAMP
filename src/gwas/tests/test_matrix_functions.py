# -*- coding: utf-8 -*-
import numpy as np
import pytest

from gwas._matrix_functions import dgesvdq, dimatcopy, set_tril, set_triu


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


@pytest.mark.parametrize("shape", [(5, 5), (5, 7), (7, 5)])
def test_set_triu(shape):
    a = np.asfortranarray(np.random.rand(*shape))
    set_triu(a)
    assert np.allclose(np.triu(a, k=1), 0)


@pytest.mark.parametrize("shape", [(5, 5), (5, 7), (7, 5)])
def test_set_tril(shape):
    a = np.asfortranarray(np.random.rand(*shape))
    set_tril(a)
    assert np.allclose(np.tril(a, k=-1), 0)


def test_dgesvdq():
    m = 1000
    n = 100

    a = np.asfortranarray(np.random.rand(m, n))
    s = np.zeros((n,))
    v = np.asfortranarray(np.zeros((n, n)))

    _, singular_values, singular_vectors = np.linalg.svd(a, full_matrices=False)

    numrank = dgesvdq(a, s, v)
    assert np.allclose(s, singular_values)
    assert np.allclose(np.abs(v), np.abs(singular_vectors))
    assert numrank == n


def test_dgesvdq_shape():
    m = 10
    n = 100

    a = np.asfortranarray(np.random.rand(m, n))
    s = np.zeros((n,))
    v = np.asfortranarray(np.zeros((n, n)))

    with pytest.raises(ValueError):
        dgesvdq(a, s, v)  # m needs to be greater or equal to n
