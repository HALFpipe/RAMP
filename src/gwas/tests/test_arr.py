# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace


def test_sa(tmp_path: Path):
    sw = SharedWorkspace.create(size=2**30)

    shape = (5, 7)
    array = sw.alloc("a", *shape)

    # include trailing
    a = array.to_numpy(include_trailing_free_memory=True)
    assert a.shape[0] == shape[0]
    assert a.shape[1] > shape[1]

    # initialize
    a = array.to_numpy()
    a[:] = np.random.rand(*shape)
    b = a.copy()

    # io
    path = tmp_path / "a.txt"
    array.to_file(path)
    array = SharedArray.from_file(path, sw)
    c = array.to_numpy()
    assert np.allclose(a, c)

    # transpose
    array.transpose()
    a = array.to_numpy()
    assert np.allclose(a, b.transpose())
    array.transpose()

    # compress
    indices = np.array([0, 3, 4])
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
