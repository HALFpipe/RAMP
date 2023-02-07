from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "mkl_trans.h":
    void mkl_dimatcopy(
        const char ordering,
        const char trans,
        size_t rows,
        size_t cols,
        const double alpha,
        double *AB,
        size_t lda,
        size_t ldb,
    )


@boundscheck(False)
@wraparound(False)
def dimatcopy(
    np.ndarray[double, ndim=2] a,
    double alpha = 1,
) -> None:
    cdef char ordering

    rows, columns = a.shape

    cdef size_t lda
    cdef size_t ldb

    if np.isfortran(a):
        ordering = "c"  # column slow row fast
        lda = a.shape[0]
        ldb = a.shape[1]
    else:
        ordering = "r"  # row slow column fast
        lda = a.shape[1]
        ldb = a.shape[0]

    mkl_dimatcopy(
        ordering,
        "t",  # transpose
        rows,
        columns,
        alpha,
        &a[0, 0],
        lda,
        ldb,
    )
