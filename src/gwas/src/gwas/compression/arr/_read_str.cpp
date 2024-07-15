#include "_check.hpp"
#include "_reader.hpp"

struct StrReader : Reader<StrReader>
{
    std::vector<PyObject *> tokens_;
    PyObject *parser_;

    PyObject *rows_;

    size_t column_indices_index_{0};
    size_t token_index_{0};

    static constexpr std::array<char, 2> delimiters = {'\n', '\t'};

    bool on_token(std::string_view token)
    {
        const size_t column_index = token_index_ % column_count_;
        token_index_++;

        if (column_indices_[column_indices_index_] == column_index)
        {
            PyObject *py_token =
                PyUnicode_DecodeASCII(token.data(), token.size(), "ignore");
            tokens_[column_indices_index_] = py_token;
            column_indices_index_++;

            if (column_indices_index_ == column_index_count_)
            {
                /* We have finished reading a record and can move on to the next. */
                PyObject *result = PyObject_Vectorcall(parser_, tokens_.data(),
                                                       tokens_.size(), nullptr);
                if (!result)
                {
                    throw error_already_set();
                }
                Py_INCREF(result);
                for (PyObject *py_token : tokens_)
                {
                    Py_DECREF(py_token);
                }

                if (result != Py_None)
                {
                    PyList_Append(rows_, result);
                }
                Py_DECREF(result);

                column_indices_index_ = 0;
            }
        }
        return true;
    }

    StrReader(PyObject *rows, PyObject *parser,
              int file_descriptor, size_t skip_bytes,
              size_t column_count, uint32_t *column_indices,
              size_t column_index_count,
              size_t ring_buffer_size)
        : Reader(file_descriptor, skip_bytes, column_count, column_indices,
                 column_index_count, ring_buffer_size),
          tokens_(column_index_count), parser_(parser), rows_(rows)
    {
    }
    virtual ~StrReader() = default;
};

static PyObject *ReadStr(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *row_list;
    PyObject *row_parser;
    int file_descriptor = 0;
    unsigned long skip_bytes = 0;
    unsigned int column_count = 0;
    PyArrayObject *column_indices_array;
    unsigned int ring_buffer_size = level1_dcache_size;

    static const char *keywords[] = {
        "row_list",
        "row_parser",
        "file_descriptor",
        "skip_bytes",
        "column_count",
        "column_indices",
        "ring_buffer_size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OikIO!|$I", const_cast<char **>(keywords),
                                     &(PyList_Type), &row_list,              // O!
                                     &row_parser,                            // O
                                     &file_descriptor,                       // i
                                     &skip_bytes,                            // k
                                     &column_count,                          // I
                                     &(PyArray_Type), &column_indices_array, // O!
                                     &ring_buffer_size))                     // I optional
    {
        return nullptr;
    }

    /* Get pointers to the arrays */
    auto c = CheckIndexArray(column_indices_array);
    if (!c)
    {
        return nullptr;
    }
    auto [column_indices, column_index_count] = *c;
    try
    {
        StrReader reader{row_list,
                         row_parser,
                         file_descriptor,
                         static_cast<size_t>(skip_bytes),
                         static_cast<size_t>(column_count),
                         column_indices,
                         column_index_count,
                         static_cast<size_t>(ring_buffer_size)};
        reader.loop();
    }
    catch (const error_already_set &e)
    {
        return nullptr;
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {.ml_name = "read_str",
     .ml_meth = _PyCFunction_CAST(ReadStr),
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "Read all specified columns from a tab-delimited file into a "
               "list of records"}};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_read_str",
    .m_doc = "A module to read strings from delimited files",
    /* Setting m_size to -1 means that the module does not support
     * sub-interpreters, because it has global state.
     */
    .m_size = -1,
    .m_methods = methods,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL};

PyMODINIT_FUNC PyInit__read(void)
{
    import_array();
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}
