# cython: language_level=3
import atexit

from cython cimport boundscheck, cast, wraparound
from libc.stdint cimport int64_t

import numpy as np

cimport numpy as np

np.import_array()
blosc2_init()


@atexit.register
def destroy():
    blosc2_destroy()


cdef extern from "blosc2.h":
    ctypedef struct blosc2_schunk:
        pass

    void blosc2_init()
    void blosc2_destroy()

    blosc2_schunk *blosc2_schunk_open(
        const char* urlpath
    ) nogil
    int blosc2_meta_exists(
        blosc2_schunk *schunk,
        const char *name
    ) nogil
    int b2nd_from_schunk(
        blosc2_schunk* schunk,
        b2nd_array_t** array
    ) nogil


cdef extern from "b2nd.h":
    ctypedef struct b2nd_array_t:
        pass

    int b2nd_get_orthogonal_selection(
        b2nd_array_t* array,
        int64_t** selection,
        int64_t* selection_size,
        void* buffer,
        int64_t* buffershape,
        int64_t buffersize
    ) nogil
    int b2nd_free(b2nd_array_t *array) nogil


@boundscheck(False)
@wraparound(False)
def get_orthogonal_selection(
    bytes urlpath,
    np.ndarray[np.int64_t, ndim=1] row_indices,
    np.ndarray[np.int64_t, ndim=1] column_indices,
    np.ndarray[np.float64_t, ndim=2, mode="c"] array,
) :
    cdef int64_t row_count = row_indices.size
    cdef int64_t column_count = column_indices.size

    if array.shape[0] != row_count:
        raise ValueError("Array row count does not match selection shape")
    if array.shape[1] != column_count:
        raise ValueError("Array column count does not match selection shape")
    if array.strides[0] != cast(int, sizeof(np.float64_t) * column_count):
        raise ValueError("Array row stride does not match dtype")
    if array.strides[1] != cast(int, sizeof(np.float64_t)):
        raise ValueError("Array column stride does not match dtype")

    cdef char *urlpath_char = urlpath

    cdef int64_t **selection = [&row_indices[0], &column_indices[0]]
    cdef int64_t *selection_shape = [row_count, column_count]
    cdef int64_t *array_shape = selection_shape
    cdef int64_t array_size = row_count * column_count

    cdef blosc2_schunk *schunk
    cdef b2nd_array_t *b2nd_array
    cdef int return_code

    with nogil:
        schunk = blosc2_schunk_open(urlpath_char)

        return_code = b2nd_from_schunk(schunk, &b2nd_array)
        if return_code < 0:
            raise RuntimeError(f"Could not create array from schunk: {return_code}")

        return_code = b2nd_get_orthogonal_selection(
            b2nd_array, selection, selection_shape, array.data, array_shape, array_size
        )
        if return_code < 0:
            raise RuntimeError(f"Could not get orthogonal selection: {return_code}")

        return_code = b2nd_free(b2nd_array)
        if return_code < 0:
            raise RuntimeError(f"Error while freeing the array: {return_code}")
