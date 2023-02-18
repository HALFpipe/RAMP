# -*- coding: utf-8 -*-
from gwas.mem.lim import memory_limit


def test_memory_limit():
    m = memory_limit()

    assert isinstance(m, int)
    assert m > 0
