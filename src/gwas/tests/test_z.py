# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from gwas.z import CompressedTextReader, CompressedTextWriter


@pytest.mark.parametrize("compression", ["zst", "xz", "gz"])
def test_z(tmp_path: Path, compression: str):
    x: str = "test" * 1000

    test_path = tmp_path / f"test.txt.{compression}"
    with CompressedTextWriter(test_path) as file_handle:
        file_handle.write(f"{x}\n")
    with CompressedTextReader(test_path) as file_handle:
        assert file_handle.read().strip() == x
