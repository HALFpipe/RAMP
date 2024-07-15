from typing import NewType, TypeVar

import numpy as np
from numpy import typing as npt

T = TypeVar("T")
FloatReader = NewType("FloatReader", object)

def create_vcf_float_reader(
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
    field_index: int,
    ring_buffer_size: int = ...,
) -> FloatReader: ...
def run_vcf_float_reader(
    data_array: npt.NDArray[np.float64],
    float_reader: FloatReader,
    row_indices: npt.NDArray[np.uint32],
) -> None: ...
def read_float(
    array: npt.NDArray[np.float64],
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
    row_indices: npt.NDArray[np.uint32],
    ring_buffer_size: int = ...,
) -> FloatReader: ...
