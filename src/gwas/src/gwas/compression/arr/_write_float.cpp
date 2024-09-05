#include <algorithm> // std::copy
#include <array>     // std::array
#include <charconv>  // std::from_chars
#include <cstddef>
#include <cstdint>   // uint32_t
#include <cstdio>    // fopen
#include <exception> // std::exception
#include <format>    // std::format
#include <iostream>  // std::cout
#include <string>
#include <string_view> // std::string_view
#include <vector>      // std::vector

#include <unistd.h>
#include <x86intrin.h>
#include <ext/stdio_filebuf.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarrayobject.h"

#include "_check.hpp"

constexpr size_t max_chars_per_value = 128;

static PyObject *WriteFloat(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *row_prefix_iterator;
    PyObject *arrays_sequence;
    int file_descriptor = 0;
    unsigned int num_threads = 1;

    static const char *keywords[] = {
        "row_prefix_generator",
        "arrays",
        "file_descriptor",
        "num_threads", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiI", const_cast<char **>(keywords),
                                     &row_prefix_iterator, // O
                                     &arrays_sequence,     // O
                                     &file_descriptor,     // i
                                     &num_threads))        // I
    {
        // PyArg_ParseTupleAndKeywords has raised an exception
        return nullptr;
    }

    if (!PyIter_Check(row_prefix_iterator))
    {
        PyErr_SetString(PyExc_TypeError, "`row_prefix_iterator` must be an iterator");
        return nullptr;
    }

    if (!PySequence_Check(arrays_sequence))
    {
        PyErr_SetString(PyExc_TypeError, "`arrays` must be a sequence");
        return nullptr;
    }

    const ssize_t array_count = PySequence_Size(arrays_sequence);
    if (array_count == -1)
    {
        // PySequence_Fast_GET_SIZE has raised an exception
        return nullptr;
    }
    if (array_count == 0)
    {
        PyErr_SetString(PyExc_ValueError, "`arrays` must have at least one array");
        return nullptr;
    }

    size_t row_count = 0;
    std::vector<PyArrayObject *> arrays(array_count, nullptr);
    for (ssize_t array_index = 0; array_index < array_count; array_index++)
    {
        PyObject *item = PySequence_GetItem(arrays_sequence, array_index);
        Py_DECREF(item);
        if (item == nullptr)
        {
            // PySequence_GetItem has raised an exception
            return nullptr;
        }
        if (!PyArray_Check(item))
        {
            auto error_string = std::format("`arrays[{}] must be a PyArrayObject`", array_index);
            PyErr_SetString(PyExc_TypeError, error_string.c_str());
            return nullptr;
        }
        PyArrayObject *array = reinterpret_cast<PyArrayObject *>(item);
        if (!CheckFloatArray(array))
        {
            return nullptr;
        }
        npy_intp *const shape = PyArray_SHAPE(array);
        if (array_index == 0)
        {
            row_count = shape[0];
        }
        else if (shape[0] != static_cast<npy_intp>(row_count))
        {
            PyErr_SetString(PyExc_ValueError, "`arrays` must have the same number of rows");
            return nullptr;
        }
        arrays[array_index] = array;
    }
    if (row_count == 0)
    {
        // Nothing to do
        Py_RETURN_NONE;
    }

    __gnu_cxx::stdio_filebuf<char> stdio_filebuf(dup(file_descriptor), std::ios::out);
    std::ostream output_stream(&stdio_filebuf);

    Py_BEGIN_ALLOW_THREADS;
    try
    {
        for (size_t row_index = 0; row_index < row_count; row_index++)
        {
            Py_BLOCK_THREADS;
            PyObject *row_prefix = PyIter_Next(row_prefix_iterator);
            if (row_prefix == nullptr)
            {
                // PyIter_Next has raised an exception
                throw error_already_set();
            }
            if (row_prefix != Py_None)
            {
                if (!PyBytes_Check(row_prefix))
                {
                    throw std::runtime_error("`row_prefix_iterator` must yield bytes objects or None");
                }
                char *row_prefix_bytes = PyBytes_AsString(row_prefix);
                if (row_prefix_bytes == nullptr)
                {
                    // PyBytes_AsString has raised an exception
                    throw error_already_set();
                }
                ssize_t row_prefix_size = PyBytes_Size(row_prefix);
                auto chars = std::string_view(row_prefix_bytes, row_prefix_bytes + row_prefix_size);
                output_stream << chars;
                Py_DECREF(row_prefix);
                output_stream << '\t';
            }
            Py_UNBLOCK_THREADS;

            bool first_column = true;
            for (auto array : arrays)
            {
                npy_intp *const shape = PyArray_SHAPE(array);
                size_t column_count = shape[1];
                for (size_t column_index = 0; column_index < column_count; column_index++)
                {
                    if (!first_column)
                    {
                        output_stream << '\t';
                    }
                    else
                    {
                        first_column = false;
                    }

                    const double value = *(static_cast<const double *> PyArray_GETPTR2(array, row_index, column_index));

                    std::array<char, max_chars_per_value> str;
                    auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), value, std::chars_format::general);
                    if (ec != std::errc())
                    {
                        std::string error_message = std::make_error_code(ec).message();
                        throw std::runtime_error(error_message.c_str());
                    }
                    auto chars = std::string_view(str.data(), ptr);

                    output_stream << chars;
                }
            }
            output_stream << '\n';
        }
    }
    catch (const error_already_set &e)
    {
        return nullptr;
    }
    catch (const std::exception &e)
    {
        Py_BLOCK_THREADS;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    Py_END_ALLOW_THREADS;

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
    .m_name = "_write_float",
    .m_doc = "A module to write structured data to delimited files",
    /* Setting m_size to -1 means that the module does not support
     * sub-interpreters, because it has global state.
     */
    .m_size = -1,
    .m_methods = methods,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL};

PyMODINIT_FUNC PyInit__write_float(void)
{
    import_array();
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}
