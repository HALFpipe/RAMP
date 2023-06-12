from typing import Callable, NewType, TypeVar

import numpy as np
from numpy import typing as npt

T = TypeVar("T")
FloatReader = NewType("FloatReader", object)

def read_str(
    row_list: list[T],
    row_parser: Callable[..., T],
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
) -> None: ...
def create_float_reader(
    file_descriptor: int,
    skip_bytes: int,
    column_count: int,
    column_indices: npt.NDArray[np.uint32],
    field_index: int,
) -> FloatReader: ...
def run_float_reader(
    data_array: npt.NDArray[np.float64],
    float_reader: FloatReader,
    row_indices: npt.NDArray[np.uint32],
) -> None: ...
