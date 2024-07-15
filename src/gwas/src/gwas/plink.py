from pathlib import Path

import numpy as np

from .compression.arr._read_str import read_str
from .compression.pipe import CompressedTextReader
from .vcf.base import read_header


def identity(variant_id: str) -> str:
    return variant_id


class ColumnReader(CompressedTextReader):
    header_length: int
    column_count: int
    column_index: int

    def __init__(
        self, file_path: Path, header_length: int, column_count: int, column_index: int
    ) -> None:
        super().__init__(file_path)

        self.header_length = header_length
        self.column_count = column_count
        self.column_index = column_index

    def read_values(self) -> list[str]:
        values: list[str] = list()
        with self as file_handle:
            read_str(
                values,
                identity,
                file_handle.fileno(),
                self.header_length,
                self.column_count,
                np.array([self.column_index], dtype=np.uint32),
            )
        return values


class HeaderColumnReader(ColumnReader):
    def __init__(self, file_path: Path, column: str) -> None:
        compressed_text_reader = CompressedTextReader(file_path)
        header_length, columns, _ = read_header(compressed_text_reader)
        column_index = columns.index(column)
        super().__init__(file_path, header_length, len(columns), column_index)


class PVarFile(HeaderColumnReader):
    def __init__(self, pfile_path: Path) -> None:
        super().__init__(pfile_path.with_suffix(".pvar"), "ID")

    def read_variant_ids(self) -> list[str]:
        return self.read_values()


class PsamFile(HeaderColumnReader):
    def __init__(self, pfile_path: Path) -> None:
        super().__init__(pfile_path.with_suffix(".psam"), "IID")

    def read_samples(self) -> list[str]:
        return self.read_values()


class BimFile(ColumnReader):
    def __init__(self, bfile_path: Path) -> None:
        super().__init__(bfile_path.with_suffix(".bim"), 0, 6, 1)

    def read_variant_ids(self) -> list[str]:
        return self.read_values()


class FamFile(ColumnReader):
    def __init__(self, bfile_path: Path) -> None:
        super().__init__(bfile_path.with_suffix(".fam"), 0, 6, 1)

    def read_samples(self) -> list[str]:
        return self.read_values()
