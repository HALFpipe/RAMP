# cython: language_level=3
import atexit

from cython cimport boundscheck, wraparound, numeric
from cython.operator cimport dereference
from libc.stdint cimport uint8_t, int16_t, int64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libcpp cimport bool as c_bool

import numpy as np

cimport numpy as np

np.import_array()
blosc2_init()


@atexit.register
def destroy():
    blosc2_destroy()


cdef extern from "blosc2.h":
    void blosc2_init()
    void blosc2_destroy()

    ctypedef struct blosc2_dparams:
        int16_t nthreads

    ctypedef struct blosc2_storage:
        blosc2_dparams* dparams

    ctypedef struct blosc2_schunk:
        blosc2_storage* storage
        blosc2_context* dctx

    ctypedef struct blosc2_io:
        uint8_t id
        const char *name
        void* params

    blosc2_schunk *blosc2_schunk_open_udio(
        const char* urlpath,
        const blosc2_io *udio
    ) nogil

    ctypedef struct blosc2_context:
        pass

    blosc2_context* blosc2_create_dctx(
        blosc2_dparams dparams
    ) nogil
    void blosc2_free_ctx(
        blosc2_context * context
    ) nogil

    int blosc2_meta_exists(
        blosc2_schunk *schunk,
        const char *name
    ) nogil

    ctypedef enum:
        BLOSC2_IO_FILESYSTEM_MMAP

    ctypedef struct blosc2_stdio_mmap:
        const char* mode
        int64_t initial_mapping_size
        c_bool needs_free

    blosc2_stdio_mmap blosc2_get_blosc2_stdio_mmap_defaults() nogil


cdef extern from "b2nd.h":
    ctypedef struct b2nd_array_t:
        pass

    int b2nd_from_schunk(
        blosc2_schunk* schunk,
        b2nd_array_t** array
    ) nogil

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
    np.ndarray[numeric, ndim=2, mode="c"] array,
    int num_threads = 1,
) :
    cdef int64_t row_count = row_indices.size
    cdef int64_t column_count = column_indices.size

    if array.shape[0] != row_count:
        raise ValueError("Array row count does not match selection shape")
    if array.shape[1] != column_count:
        raise ValueError("Array column count does not match selection shape")

    cdef char *urlpath_char = urlpath

    cdef int64_t **selection = [&row_indices[0], &column_indices[0]]
    cdef int64_t *selection_shape = [row_count, column_count]
    cdef int64_t *array_shape = selection_shape
    cdef int64_t array_size = row_count * column_count

    cdef blosc2_stdio_mmap default_params
    cdef blosc2_stdio_mmap* params
    cdef blosc2_schunk *schunk
    cdef b2nd_array_t *b2nd_array
    cdef int return_code

    with nogil:
        # schunk = blosc2_schunk_open(urlpath_char)

        params = <blosc2_stdio_mmap *> malloc(sizeof(blosc2_stdio_mmap))
        default_params = blosc2_get_blosc2_stdio_mmap_defaults()
        memcpy(params, &default_params, sizeof(blosc2_stdio_mmap))
        params.mode = b"r"
        params.needs_free = True

        io = <blosc2_io *> malloc(sizeof(blosc2_io))
        io.id = BLOSC2_IO_FILESYSTEM_MMAP
        io.params = params

        schunk = blosc2_schunk_open_udio(urlpath_char, io)
        free(io)

        schunk.storage.dparams.nthreads = num_threads
        blosc2_free_ctx(schunk.dctx)
        schunk.dctx = blosc2_create_dctx(dereference(schunk.storage.dparams))
        if schunk.dctx == NULL:
            raise RuntimeError("Could not create decompression context")

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
