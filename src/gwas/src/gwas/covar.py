import warnings

import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt
from upath import UPath

from ._matrix_functions import copy_triu_tril
from .compression.arr.base import CompressionMethod, FileArray, FileArrayWriter
from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace
from .pheno import VariableCollection


def calc_and_save_covariance(
    vc: VariableCollection,
    path: UPath,
    compression_method: CompressionMethod,
    num_threads: int,
) -> UPath:
    path = path.with_suffix(compression_method.suffix)
    if path.is_file():
        logger.debug("Skip writing covariance matrix because it already exists")
        return path

    names = vc.names
    covariance_array = calc_covariance(vc.sw, vc.to_numpy())
    covariance = covariance_array.to_numpy()

    writer: FileArrayWriter[np.float64] = FileArray.create(
        path,
        covariance.shape,
        covariance.dtype.type,
        compression_method,
        num_threads=num_threads,
    )

    data_frame = pd.DataFrame(dict(variable=names))
    writer.set_axis_metadata(0, data_frame)
    writer.set_axis_metadata(1, names)

    with writer:
        writer[:, :] = covariance

    covariance_array.free()

    return writer.file_path


def calc_covariance(
    sw: SharedWorkspace, data: npt.NDArray[np.float64]
) -> SharedArray[np.float64]:
    row_count, variable_count = data.shape

    logger.debug("Calculating covariance matrix")
    degrees_of_freedom_array = sw.alloc(
        "degrees-of-freedom", variable_count, variable_count
    )
    degrees_of_freedom = degrees_of_freedom_array.to_numpy()
    covariance_array = sw.alloc("covariance", variable_count, variable_count)
    covariance = covariance_array.to_numpy()
    a_array = sw.alloc("a", row_count, variable_count)
    a = a_array.to_numpy()

    # Subtract one from counts to get degrees of freedom
    np.isfinite(data, out=a)
    scipy.linalg.blas.dsyrk(
        alpha=1.0, a=a, trans=1, c=degrees_of_freedom, overwrite_c=True
    )
    np.subtract(degrees_of_freedom, 1.0, out=degrees_of_freedom)

    # Set lower triangle to 1 to avoid division by zero
    degrees_of_freedom[degrees_of_freedom <= 0.0] = 1.0

    a[:] = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a -= np.nanmean(a, axis=0)

    a = np.nan_to_num(a, copy=False)

    scipy.linalg.blas.dsyrk(alpha=1.0, a=a, trans=1, c=covariance, overwrite_c=True)
    a_array.free()

    np.divide(covariance, degrees_of_freedom, out=covariance)
    degrees_of_freedom_array.free()

    copy_triu_tril(covariance)
    return covariance_array
