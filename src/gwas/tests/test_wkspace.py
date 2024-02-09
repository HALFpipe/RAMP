# -*- coding: utf-8 -*-
import numpy as np
import pytest
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from numpy import typing as npt


def test_sw_merge():
    sw = SharedWorkspace.create()

    names = list()
    for i in range(10):
        name = f"m-{i}"

        sw.alloc(name, 1000, 100)
        a = sw.get_array(name).to_numpy()
        a[1, 7] = 10

        names.append(name)

    sw.merge(*names)

    a = sw.get_array("m-0").to_numpy()
    assert np.all(a[1, 7::100])
    assert np.isclose(a.sum(), 100)

    sw.close()
    sw.unlink()


def test_sw_mem():
    sw = SharedWorkspace.create(size=2**21)

    with pytest.raises(MemoryError):
        sw.alloc("l", 1000, 1000)

    sw.close()
    sw.unlink()


def test_sw_squash():
    sw = SharedWorkspace.create()

    n = 100

    arrays: list[SharedArray] = list()
    for i in range(7):
        name = f"s-{i}"

        array = sw.alloc(name, n, i + 2)
        a = array.to_numpy()
        a[:] = np.random.rand(*a.shape)

        arrays.append(array)

    numpy_arrays: list[npt.NDArray] = list()
    for i in range(7):
        if i % 2 == 0:
            numpy_arrays.append(arrays[i].to_numpy().copy())
        else:
            sw.free(arrays[i].name)

    sw.squash()

    names = [a.name for a in arrays if a.name in sw.allocations]
    array = sw.merge(*names)

    numpy_array = np.hstack(numpy_arrays)

    assert np.allclose(array.to_numpy(), numpy_array)

    sw.close()
    sw.unlink()
