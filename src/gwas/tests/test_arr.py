import numpy as np
import pytest
from pytest import FixtureRequest
from upath import UPath

from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.utils.multiprocessing import get_global_lock

shape = (5, 7)


@pytest.fixture(scope="function")
def shared_array(request: FixtureRequest, sw: SharedWorkspace) -> SharedArray:
    with get_global_lock():
        shared_array = sw.alloc(SharedArray.get_name(sw), *shape)
    request.addfinalizer(shared_array.free)

    a = shared_array.to_numpy()
    a[:] = np.random.rand(*shape)

    return shared_array


def test_include_trailing(shared_array: SharedArray) -> None:
    a = shared_array.to_numpy(include_trailing_free_memory=True)
    assert a.shape[0] == shape[0]
    assert a.shape[1] > shape[1]


def test_io(shared_array: SharedArray, tmp_path: UPath) -> None:
    path = tmp_path / "a"
    path = shared_array.to_file(path)

    array = SharedArray.from_file(path, shared_array.sw, np.float64)

    a = shared_array.to_numpy()
    c = array.to_numpy()
    np.testing.assert_allclose(a, c)


def test_transpose(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    shared_array.transpose()
    a = shared_array.to_numpy()
    np.testing.assert_allclose(a, b.transpose())


def test_compress_rows(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    indices = np.array([0, 3, 4], dtype=np.uint32)
    shared_array.compress(indices)
    a = shared_array.to_numpy()
    np.testing.assert_allclose(b[indices, :], a)


def test_compress_columns(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    indices = np.array([0, 3, 4], dtype=np.uint32)
    shared_array.compress(indices, axis=1)
    a = shared_array.to_numpy()
    np.testing.assert_allclose(b[:, indices], a)


def test_resize(shared_array: SharedArray) -> None:
    b = shared_array.to_numpy().copy()

    m = min(shape)
    shared_array.resize(m, m)

    a = shared_array.to_numpy()
    assert a.shape == (m, m)
    np.testing.assert_allclose(b[:, :m], a)
