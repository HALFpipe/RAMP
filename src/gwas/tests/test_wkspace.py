# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy import typing as npt

from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace


def test_sw_merge() -> None:
    sw = SharedWorkspace.create()

    arrays: list[SharedArray] = list()
    for i in range(10):
        name = f"m-{i}"

        sw.alloc(name, 1000, 100)
        a: SharedArray[np.float64] = SharedArray(name, sw)
        a.to_numpy()[1, 7] = 10

        arrays.append(a)

    SharedArray.merge(*arrays)

    a = SharedArray("m-0", sw)
    assert np.all(a.to_numpy()[1, 7::100])
    assert np.isclose(a.to_numpy().sum(), 100)

    sw.close()
    sw.unlink()


def test_sw_mem() -> None:
    sw = SharedWorkspace.create(size=2**21)

    with pytest.raises(MemoryError):
        sw.alloc("l", 1000, 1000)

    sw.close()
    sw.unlink()


def test_sw_squash() -> None:
    sw = SharedWorkspace.create()

    n = 100

    arrays: list[SharedArray] = list()
    for i in range(7):
        name = f"s-{i}"

        array = sw.alloc(name, n, i + 2)
        a = array.to_numpy()
        a[:] = np.random.rand(*a.shape)

        arrays.append(array)

    numpy_arrays: list[npt.NDArray[np.float64]] = list()
    for i in range(7):
        if i % 2 == 0:
            numpy_arrays.append(arrays[i].to_numpy().copy())
        else:
            sw.free(arrays[i].name)

    sw.squash()

    arrays = [a for a in arrays if a.name in sw.allocations]
    array = SharedArray.merge(*arrays)

    numpy_array = np.hstack(numpy_arrays)

    assert np.allclose(array.to_numpy(), numpy_array)

    sw.close()
    sw.unlink()
