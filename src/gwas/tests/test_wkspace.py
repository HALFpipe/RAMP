# -*- coding: utf-8 -*-
import numpy as np

from gwas.wkspace import SharedWorkspace


def test_shared_workspace():
    sw = SharedWorkspace.create()

    names = list()
    for i in range(10):
        name = f"test-{i}"

        sw.alloc(name, 1000, 100)
        a = sw.get_array(name)
        a[1, 7] = 10

        names.append(name)

    sw.merge(*names)

    a = sw.get_array("test-0")
    assert np.all(a[1, 7::100])
    assert np.isclose(a.sum(), 100)

    sw.close()
    sw.unlink()
