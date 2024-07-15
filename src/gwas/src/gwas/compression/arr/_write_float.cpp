#include <algorithm> // std::copy
#include <array>     // std::array
#include <charconv>  // std::from_chars
#include <cstddef>
#include <cstdint>   // uint32_t
#include <exception> // std::exception
#include <format>    // std::format
#include <iostream>  // std::cout
#include <string>
#include <string_view> // std::string_view
#include <vector>      // std::vector

#include <unistd.h>
#include <x86intrin.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarrayobject.h"

#include "_check.hpp"

static PyObject *WriteFloat(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // std::cout << "ReadStr" << std::endl;

    PyObject *row_prefix_iterator;
    PyArrayObject *array;
    int file_descriptor = 0;

    static const char *keywords[] = {
        "row_prefix_generator",
        "array",
        "file_descriptor", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO!i", const_cast<char **>(keywords),
                                     &row_prefix_iterator,    // O
                                     &(PyArray_Type), &array, // O!
                                     &file_descriptor))       // i
    {
        // PyArg_ParseTupleAndKeywords has raised an exception
        return nullptr;
    }

    if (!PyIter_Check(row_prefix_iterator))
    {
        PyErr_SetString(PyExc_TypeError, "`row_prefix_iterator` must be an iterator");
        return nullptr;
    }

    if (!CheckFloatArray(array))
    {
        return nullptr;
    }
    npy_intp *shape = PyArray_SHAPE(array);
    size_t row_count = shape[0];
    size_t column_count = shape[1];

    FILE *file = fdopen(dup(file_descriptor), "a");
    std::array<char, 32> str;
    try
    {
        for (size_t row_index = 0; row_index < row_count; row_index++)
        {
            PyObject *row_prefix = PyIter_Next(row_prefix_iterator);
            if (row_prefix == nullptr)
            {
                // PyIter_Next has raised an exception
                return nullptr;
            }
            if (!PyBytes_Check(row_prefix))
            {
                PyErr_SetString(PyExc_TypeError, "`row_prefix_iterator` must yield bytes objects");
                return nullptr;
            }
            Py_ssize_t row_prefix_size = PyBytes_Size(row_prefix);
            char *row_prefix_bytes = PyBytes_AsString(row_prefix);
            if (row_prefix_bytes == nullptr)
            {
                // PyBytes_AsString has raised an exception
                return nullptr;
            }
            fwrite(row_prefix_bytes, 1, row_prefix_size, file);
            Py_DECREF(row_prefix);

            for (size_t column_index = 0; column_index < column_count; column_index++)
            {
                if (fwrite("\t", 1, 1, file) != 1)
                {
                    PyErr_SetString(PyExc_IOError, "Failed to write to file");
                    return nullptr;
                }

                const double value = *reinterpret_cast<const double *> PyArray_GETPTR2(array, row_index, column_index);

                auto [ptr, ec] = std::to_chars(str.data(), str.end(), value, std::chars_format::general);
                if (ec != std::errc())
                {
                    std::string error_message = std::make_error_code(ec).message();
                    PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
                    return nullptr;
                }
                size_t length = ptr - str.data();
                if (fwrite(str.begin(), 1, length, file) != length)
                {
                    PyErr_SetString(PyExc_IOError, "Failed to write to file");
                    return nullptr;
                }
            }
            if (fwrite("\n", 1, 1, file) != 1)
            {
                PyErr_SetString(PyExc_IOError, "Failed to write to file");
                return nullptr;
            }
        }
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    if (fclose(file) != 0)
    {
        PyErr_SetString(PyExc_IOError, "Failed to close file");
        return nullptr;
    }

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {.ml_name = "write_float",
     .ml_meth = _PyCFunction_CAST(WriteFloat),
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "Read all specified columns from a tab-delimited file into a "
               "list of records"},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_write",
    .m_doc = "A module to write structured data to delimited files",
    /* Setting m_size to -1 means that the module does not support
     * sub-interpreters, because it has global state.
     */
    .m_size = -1,
    .m_methods = methods,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL};

PyMODINIT_FUNC PyInit__write(void)
{
    import_array();
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}
