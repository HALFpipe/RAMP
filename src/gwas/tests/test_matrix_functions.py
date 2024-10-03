import numpy as np
import pytest

from gwas._matrix_functions import dgesvdq, dimatcopy, set_tril, set_triu


def test_dimatcopy_square() -> None:
    a = np.random.rand(8, 8)

    b = a.copy()
    dimatcopy(b)

    np.testing.assert_allclose(a.transpose(), b)


def test_dimatcopy_rect() -> None:
    a = np.random.rand(5, 8)

    b = a.copy()
    dimatcopy(b)

    np.testing.assert_allclose(
        a,
        np.ndarray(b.shape, b.dtype, buffer=b.data, order="F"),
    )


@pytest.mark.parametrize("shape", [(5, 5), (5, 7), (7, 5)])
def test_set_triu(shape: tuple[int, ...]) -> None:
    a = np.asfortranarray(np.random.rand(*shape))
    set_triu(a)
    np.testing.assert_allclose(np.triu(a, k=1), 0)


@pytest.mark.parametrize("shape", [(5, 5), (5, 7), (7, 5)])
def test_set_tril(shape: tuple[int, ...]) -> None:
    a = np.asfortranarray(np.random.rand(*shape))
    set_tril(a)
    np.testing.assert_allclose(np.tril(a, k=-1), 0)


def test_dgesvdq() -> None:
    m = 1000
    n = 100

    a = np.asfortranarray(np.random.rand(m, n))
    s = np.zeros((n,))
    v = np.asfortranarray(np.zeros((n, n)))

    _, singular_values, singular_vectors = np.linalg.svd(a, full_matrices=False)

    numrank = dgesvdq(a, s, v)
    np.testing.assert_allclose(s, singular_values)
    np.testing.assert_allclose(np.abs(v), np.abs(singular_vectors))
    assert numrank == n


def test_dgesvdq_shape() -> None:
    m = 10
    n = 100

    a = np.asfortranarray(np.random.rand(m, n))
    s = np.zeros((n,))
    v = np.asfortranarray(np.zeros((n, n)))

    with pytest.raises(ValueError):
        dgesvdq(a, s, v)  # m needs to be greater or equal to n
