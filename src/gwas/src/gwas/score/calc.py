# -*- coding: utf-8 -*-
import numpy as np
from numpy import typing as npt

from ..log import logger


def calc_u_stat(
    inverse_variance_scaled_residuals: npt.NDArray,
    rotated_genotypes: npt.NDArray,
    u_stat: npt.NDArray,
) -> None:
    logger.debug("Calculating numerator")
    u_stat[:] = rotated_genotypes.transpose() @ inverse_variance_scaled_residuals


def calc_v_stat(
    inverse_variance: npt.NDArray,
    squared_genotypes: npt.NDArray,
    u_stat: npt.NDArray,
    v_stat: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    logger.debug("Calculating denominator")
    v_stat[:] = squared_genotypes.transpose() @ inverse_variance
    logger.debug("Zeroing invalid values")
    invalid = np.isclose(v_stat, 0)
    u_stat[invalid] = 0
    v_stat[invalid] = 1
    return invalid
