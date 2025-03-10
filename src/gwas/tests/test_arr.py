import numpy as np
import pytest
from upath import UPath

from gwas.mem.arr import SharedArray, invert_pivot
from gwas.mem.wkspace import SharedWorkspace
from gwas.utils.multiprocessing import get_global_lock

from .utils import check_memory_leaks


@pytest.fixture(scope="function", params=[(5, 7), (5000, 7000)])
def shared_array(request: pytest.FixtureRequest, sw: SharedWorkspace) -> SharedArray:
    rng = np.random.default_rng(0)

    shape = request.param
    with get_global_lock():
        shared_array = sw.alloc(SharedArray.get_name(sw), *shape)
    request.addfinalizer(shared_array.free)

    a = shared_array.to_numpy()
    for i in range(shape[0]):
        a[i, :] = rng.normal(size=shape[1])

    return shared_array


def test_include_trailing(shared_array: SharedArray) -> None:
    a = shared_array.to_numpy(include_trailing_free_memory=True)
    assert a.shape[0] == shared_array.shape[0]
    assert a.shape[1] > shared_array.shape[1]


def test_io(shared_array: SharedArray, tmp_path: UPath) -> None:
    path = tmp_path / "a"
    path = shared_array.to_file(path)

    array = SharedArray.from_file(path, shared_array.sw, np.float64)

    a = shared_array.to_numpy()
    c = array.to_numpy()
    np.testing.assert_allclose(a, c)


def test_transpose(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    with check_memory_leaks():
        shared_array.transpose()
    a = shared_array.to_numpy()
    np.testing.assert_allclose(a, b.transpose())


def test_compress_rows(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    indices = np.array([0, 3, 4], dtype=np.uint32)
    with check_memory_leaks():
        shared_array.compress(indices)
    a = shared_array.to_numpy()
    np.testing.assert_allclose(b[indices, :], a)


def test_compress_columns(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    indices = np.array([0, 3, 4], dtype=np.uint32)
    with check_memory_leaks():
        shared_array.compress(indices, axis=1)
    a = shared_array.to_numpy()
    np.testing.assert_allclose(b[:, indices], a)


def test_resize(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    m = min(shared_array.shape)
    shared_array.resize(m, m)

    a = shared_array.to_numpy()
    assert a.shape == (m, m)
    np.testing.assert_allclose(b[:, :m], a)


def test_apply_inverse_pivot(shared_array: SharedArray) -> None:
    pivot_size = shared_array.shape[1] * 4
    with check_memory_leaks(target=pivot_size):
        pivot = shared_array.triangularize(pivoting=True)

    a = shared_array.to_numpy().copy()
    a = a[:, invert_pivot(pivot)]

    with check_memory_leaks():
        shared_array.apply_inverse_pivot(pivot)

    b = shared_array.to_numpy()
    np.testing.assert_allclose(b, a)
