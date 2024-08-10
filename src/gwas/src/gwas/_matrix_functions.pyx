# cython: language_level=3
from cython cimport boundscheck, cast, wraparound

import numpy as np

cimport numpy as np

np.import_array()


def check_strides(
    np.ndarray[double, ndim=2, mode="fortran"] matrix,
    size_t n
):
    if matrix.strides[0] != cast(int, sizeof(double)):
        raise ValueError(
            f"Invalid strides for axis 0: {matrix.strides[0]} != {sizeof(double)}"
        )
    if matrix.strides[1] != cast(int, sizeof(double) * n):
        raise ValueError(
            f"Invalid strides for axis 1: {matrix.strides[1]} != {sizeof(double) * n}"
        )


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

    cdef size_t lda
    cdef size_t ldb

    if np.isfortran(a):
        ordering = b"c"  # column slow row fast
        lda = a.shape[0]
        ldb = a.shape[1]
    else:
        ordering = b"r"  # row slow column fast
        lda = a.shape[1]
        ldb = a.shape[0]

    mkl_dimatcopy(
        ordering,
        b"t",  # transpose
        a.shape[0],
        a.shape[1],
        alpha,
        &a[0, 0],
        lda,
        ldb,
    )


cdef extern:
    void c_set_lower_triangle(
        double *a,
        double alpha,
        size_t n,
        size_t m,
    )
    void c_set_upper_triangle(
        double *a,
        double alpha,
        size_t n,
        size_t m,
    )


def set_tril(
    np.ndarray[double, ndim=2, mode="fortran"] a,
    double alpha = 0,
) -> None:
    cdef size_t n = a.shape[0]
    check_strides(a, n)

    cdef size_t m = a.shape[1]

    c_set_lower_triangle(&a[0, 0], alpha, n, m)


def set_triu(
    np.ndarray[double, ndim=2, mode="fortran"] a,
    double alpha = 0,
) -> None:
    cdef size_t n = a.shape[0]
    check_strides(a, n)

    cdef size_t m = a.shape[1]

    c_set_upper_triangle(&a[0, 0], alpha, n, m)


cdef extern from "mkl_lapacke.h":
    cdef int LAPACK_COL_MAJOR
    # int LAPACKE_dgesvd(
    #     int matrix_layout,
    #     char jobu,
    #     char jobvt,
    #     int m,
    #     int n,
    #     double* a,
    #     int lda,
    #     double* s,
    #     double* u,
    #     int ldu,
    #     double* vt,
    #     int ldvt,
    #     double* superb
    # )
    int LAPACKE_dgesvdq(
        int matrix_layout,
        char joba,
        char jobp,
        char jobr,
        char jobu,
        char jobv,
        int m,
        int n,
        double* a,
        int lda,
        double* s,
        double* u,
        int ldu,
        double* v,
        int ldv,
        int* numrank
    )


# def dgesvd(
#     np.ndarray[double, ndim=2, mode="fortran"] a,
#     np.ndarray[double, ndim=1, mode="fortran"] s,
#     np.ndarray[double, ndim=2, mode="fortran"] v,
# ):
#     cdef int m = a.shape[0]
#     check_strides(a, m)

#     cdef int n = a.shape[1]
#     check_strides(v, n)

#     cdef char joba = b"H"  # Do not truncate
#     cdef char jobp = b"P"  # Enable row pivoting
#     cdef char jobr = b"N"  # No additional transpose
#     cdef char jobu = b"N"  # No left singular vectors
#     cdef char jobvt = b"A"  # All right singular vectors

#     cdef int numrank = 0

#     info = LAPACKE_dgesvd(
#         LAPACK_COL_MAJOR,
#         jobu,
#         jobvt,
#         m,
#         n,
#         &a[0, 0],
#         m,  # lda
#         &s[0],
#         NULL,  # left singular vectors
#         m,  # ldu
#         &v[0, 0],
#         n,  # ldv
#         &numrank,
#     )

#     if info != 0:
#         raise ValueError

#     return numrank


def dgesvdq(
    np.ndarray[double, ndim=2, mode="fortran"] a,
    np.ndarray[double, ndim=1, mode="fortran"] s,
    np.ndarray[double, ndim=2, mode="fortran"] v,
):
    cdef int m = a.shape[0]
    check_strides(a, m)

    cdef int n = a.shape[1]
    check_strides(v, n)

    cdef char joba = b"H"  # Do not truncate
    cdef char jobp = b"P"  # Enable row pivoting
    cdef char jobr = b"N"  # No additional transpose
    cdef char jobu = b"N"  # No left singular vectors
    cdef char jobv = b"A"  # All right singular vectors

    cdef int numrank = 0

    info = LAPACKE_dgesvdq(
        LAPACK_COL_MAJOR,
        joba,
        jobp,
        jobr,
        jobu,
        jobv,
        m,
        n,
        &a[0, 0],
        m,  # lda
        &s[0],
        NULL,  # left singular vectors
        m,  # ldu
        &v[0, 0],
        n,  # ldv
        &numrank,
    )

    if info != 0:
        raise ValueError

    return numrank
