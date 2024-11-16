from typing import NewType

import numpy as np
from numpy import typing as npt

TSVFloatReader = NewType("TSVFloatReader", object)
VCFFloatReader = NewType("VCFFloatReader", object)

def create_tsv_float_reader(
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
    ring_buffer_size: int = ...,
) -> TSVFloatReader: ...
def run_tsv_float_reader(
    data_array: npt.NDArray[np.float64],
    reader: TSVFloatReader,
    row_indices: npt.NDArray[np.uint32],
) -> int: ...
def create_vcf_float_reader(
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
    field_index: int,
    ring_buffer_size: int = ...,
) -> VCFFloatReader: ...
def run_vcf_float_reader(
    array: npt.NDArray[np.float64],
    reader: VCFFloatReader,
    row_indices: npt.NDArray[np.uint32],
) -> None: ...
