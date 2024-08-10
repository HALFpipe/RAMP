import numpy as np
from numpy import typing as npt

def get_orthogonal_selection(
    urlpath: bytes,
    row_indices: npt.NDArray[np.int64],
    column_indices: npt.NDArray[np.int64],
    array: npt.NDArray,
    num_threads: int = 1,
) -> None: ...
