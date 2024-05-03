# -*- coding: utf-8 -*-
import numpy as np
from numpy import typing as npt

from ..log import logger


def calc_u_stat(
    scaled_residuals: npt.NDArray[np.float64],
    rotated_genotypes: npt.NDArray[np.float64],
    u_stat: npt.NDArray[np.float64],
) -> None:
    logger.debug("Calculating numerator")
    u_stat[:] = rotated_genotypes.transpose() @ scaled_residuals


def calc_v_stat(
    inverse_variance: npt.NDArray[np.float64],
    squared_genotypes: npt.NDArray[np.float64],
    u_stat: npt.NDArray[np.float64],
    v_stat: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    logger.debug("Calculating denominator")
    v_stat[:] = squared_genotypes.transpose() @ inverse_variance
    logger.debug("Zeroing invalid values")
    invalid = np.isclose(v_stat, 0)
    u_stat[invalid] = 0
    v_stat[invalid] = 1
    return invalid
