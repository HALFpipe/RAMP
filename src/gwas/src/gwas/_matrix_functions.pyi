import numpy as np
from numpy import typing as npt

def dimatcopy(
    a: npt.NDArray[np.float64],
    alpha: float = 1,
) -> None: ...
def set_tril(
    a: npt.NDArray[np.float64],
    alpha: float = 0,
) -> None: ...
def set_triu(
    a: npt.NDArray[np.float64],
    alpha: float = 0,
) -> None: ...
def dgesvdq(
    a: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
) -> int: ...
