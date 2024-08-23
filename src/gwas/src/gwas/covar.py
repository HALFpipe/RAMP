import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt
from upath import UPath

from .compression.arr.base import CompressionMethod, FileArray, FileArrayWriter
from .log import logger
from .pheno import VariableCollection


def calc_covariance(
    vc: VariableCollection,
    path: UPath,
    compression_method: CompressionMethod,
    num_threads: int,
) -> UPath:
    path = path.with_suffix(compression_method.suffix)
    if path.is_file():
        logger.debug("Skip writing covariance matrix because it already exists")
        return path

    array = vc.to_numpy()
    names = [*vc.phenotype_names, *vc.covariate_names]

    logger.debug("Calculating covariance matrix")
    count = scipy.linalg.blas.dsyrk(alpha=1.0, a=np.isfinite(array), trans=1)

    # Subtract one to get degrees of freedom
    degrees_of_freedom = count - 1

    # Set lower triangle to 1 to avoid division by zero
    x, y = np.tril_indices_from(count, k=-1)
    degrees_of_freedom[(x, y)] = 1

    a = np.nan_to_num(array - np.nanmean(array, axis=0))
    product = scipy.linalg.blas.dsyrk(alpha=1.0, a=a, trans=1)

    covariance: npt.NDArray[np.float64] = product / degrees_of_freedom
    covariance[(x, y)] = covariance[(y, x)]
    covariance = np.asfortranarray(covariance)

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

    return writer.file_path
