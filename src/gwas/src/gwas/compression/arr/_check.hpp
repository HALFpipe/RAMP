#pragma once

#include <optional> // std::optional

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarrayobject.h"

inline std::optional<std::pair<uint32_t *, size_t>> CheckIndexArray(PyArrayObject *array)
{
    if (PyArray_NDIM(array) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Index arrays must be a one-dimensional");
        return std::nullopt;
    }

    if (PyArray_TYPE(array) != NPY_UINT32)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Index arrays must be an array of unsigned integers");
        return std::nullopt;
    }

    if (PyArray_STRIDES(array)[0] != sizeof(unsigned int))
    {
        PyErr_SetString(PyExc_ValueError,
                        "Index arrays must be a contiguous array");
        return std::nullopt;
    }

    size_t size = PyArray_SIZE(array);
    uint32_t *data =
        reinterpret_cast<uint32_t *>(PyArray_DATA(array));
    return std::pair(data, size);
}

inline bool CheckFloatArray(PyArrayObject *array)
{
    if (PyArray_NDIM(array) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "`array` must be a 2D array");
        return false;
    }
    if (PyArray_TYPE(array) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError,
                        "`array` must be an array of doubles");
        return false;
    }

    return true;
}
