import numpy as np
from upath import UPath

from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace


def test_sa(tmp_path: UPath) -> None:
    sw = SharedWorkspace.create(size=2**30)

    shape = (5, 7)
    array = sw.alloc("a", *shape)
    assert isinstance(array, SharedArray)

    # include trailing
    a = array.to_numpy(include_trailing_free_memory=True)
    assert a.shape[0] == shape[0]
    assert a.shape[1] > shape[1]

    # initialize
    a = array.to_numpy()
    a[:] = np.random.rand(*shape)
    b = a.copy()

    # io
    path = tmp_path / "a"
    path = array.to_file(path)
    array = SharedArray.from_file(path, sw, np.float64)
    c = array.to_numpy()
    assert np.allclose(a, c)

    # transpose
    array.transpose()
    a = array.to_numpy()
    assert np.allclose(a, b.transpose())
    array.transpose()

    # compress
    indices = np.array([0, 3, 4], dtype=np.uint32)
    array.compress(indices)
    a = array.to_numpy()
    assert np.allclose(b[indices, :], a)
    b = b[indices, :]

    # resize
    m = len(indices)
    array.resize(m, m)
    a = array.to_numpy()
    assert a.shape == (m, m)
    assert np.allclose(b[:, :m], a)

    sw.close()
    sw.unlink()
