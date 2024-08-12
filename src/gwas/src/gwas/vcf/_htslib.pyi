from typing import Callable, TypeVar

import numpy as np
from numpy import typing as npt

T = TypeVar("T")

def read_variants(
    file_path: str,
    variant_parser: Callable[[str, int, str, str, bool, float, float, float], T],
) -> tuple[list[T], list[str]]: ...
def read_dosages(
    file_path: str,
    dosages: npt.NDArray[np.float64],
    sample_indices: npt.NDArray[np.uint32],
    variant_indices: npt.NDArray[np.uint32],
) -> None: ...
