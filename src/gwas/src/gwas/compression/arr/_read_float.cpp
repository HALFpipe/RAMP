#include <charconv> // std::from_chars
#include <iostream> // std::cout

#include "_check.hpp"
#include "_reader.hpp"

template <class H>
class FloatReader : public Reader<FloatReader<H>>
{
public:
  static constexpr const char name[] = "float_reader";
  FloatReader(PyArrayObject *array,
              int file_descriptor, size_t skip_bytes, size_t column_count,
              uint32_t *column_indices, size_t column_index_count,
              uint32_t *row_indices, size_t row_index_count,
              size_t ring_buffer_size)
      : Reader<FloatReader<H>>(file_descriptor, skip_bytes, column_count,
                               column_indices, column_index_count,
                               ring_buffer_size),
        array_(array),
        row_indices_(row_indices), row_index_count_(row_index_count) {}
  virtual ~FloatReader() = default;

  static constexpr std::array<char, 2> delimiters = {'\n', '\t'};

  PyArrayObject *array_;

  uint32_t *row_indices_;
  size_t row_index_count_;
  size_t row_indices_index_{0};

  bool on_token(std::string_view token)
  {
    const unsigned int current_row_index = token_index_ / this->column_count_;
    const unsigned int current_column_index = token_index_ % this->column_count_;
    /* Always increment the token counter. */
    token_index_++;

    if (row_indices_[row_indices_index_] == current_row_index)
    {
      if (this->column_indices_[column_indices_index_] == current_column_index)
      {
        token = reinterpret_cast<H *>(this)->prepare_token(token);

        double value;
        if (std::from_chars(token.begin(), token.end(), value).ec !=
            std::errc{})
        {
          auto error_string = std::format("Could not parse float \"{}\"", token);
          throw std::runtime_error(error_string);
        }

        void *pointer = PyArray_GETPTR2(array_, row_indices_index_, column_indices_index_);
        *static_cast<double *>(pointer) = value;

        // void *array_pointer = PyArray_DATA(array_);
        // size_t offset = reinterpret_cast<size_t>(pointer) - reinterpret_cast<size_t>(array_pointer);
        // std::cout << std::format("Setting value {} at indices {}, {} at pointer {} ({} from array start {})", value, row_indices_index_, column_indices_index_, pointer, offset, array_pointer) << std::endl;

        column_indices_index_++;
        if (column_indices_index_ == this->column_index_count_)
        {
          /* We have finished reading a record and can move on to the next. */
          column_indices_index_ = 0;
          row_indices_index_++;

          if (row_indices_index_ == row_index_count_)
          {
            /* We have finished reading all records. */
            return false;
          }
        }
      }
    }
    return true;
  }

private:
  size_t token_index_{0};
  size_t column_indices_index_{0};
};

template <class T>
PyCapsule_Destructor DestroyFloatReader = [](PyObject *float_reader_capsule)
{
  if (!PyCapsule_IsValid(float_reader_capsule, FloatReader<T>::name))
  {
    return;
  }

  FloatReader<T> *float_reader = (FloatReader<T> *)PyCapsule_GetPointer(
      float_reader_capsule, FloatReader<T>::name);
  if (float_reader)
  {
    delete float_reader;
  }
};

struct TSVFloatReader : FloatReader<TSVFloatReader>
{
  TSVFloatReader(PyArrayObject *array,
                 int file_descriptor, size_t skip_bytes, size_t column_count,
                 uint32_t *column_indices, size_t column_index_count,
                 uint32_t *row_indices, size_t row_index_count,
                 size_t ring_buffer_size)
      : FloatReader(array,
                    file_descriptor, skip_bytes, column_count,
                    column_indices, column_index_count,
                    row_indices, row_index_count,
                    ring_buffer_size) {}
  virtual ~TSVFloatReader() = default;

  std::string_view prepare_token(std::string_view token)
  {
    return token;
  }
};

static PyObject *CreateTSVFloatReader(PyObject *self, PyObject *args, PyObject *kwargs)
{
  int file_descriptor = 0;
  unsigned long skip_bytes = 0;
  unsigned int column_count = 0;
  PyArrayObject *column_indices_array;
  unsigned int ring_buffer_size = default_ring_buffer_size;

  static const char *keywords[] = {
      "file_descriptor",
      "skip_bytes",
      "column_count",
      "column_indices",
      "ring_buffer_size", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ikIO!|$I", const_cast<char **>(keywords),
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

  TSVFloatReader *reader = nullptr;
  try
  {
    reader = new TSVFloatReader(
        nullptr, file_descriptor, static_cast<size_t>(skip_bytes),
        static_cast<size_t>(column_count), column_indices, column_index_count,
        nullptr, 0, static_cast<size_t>(ring_buffer_size));
  }
  catch (const std::exception &e)
  {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  if (!reader)
  {
    PyErr_SetString(PyExc_ValueError, "Failure creating FloatReader");
    return nullptr;
  }
  PyObject *capsule =
      PyCapsule_New(reader, TSVFloatReader::name, DestroyFloatReader<TSVFloatReader>);

  if (!capsule)
  {
    PyErr_SetString(PyExc_ValueError, "Failure creating PyCapsule");
    return nullptr;
  }

  return capsule;
}

static PyObject *RunTSVFloatReader(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyArrayObject *array;
  PyObject *reader_capsule;
  PyArrayObject *row_indices_array;

  static const char *keywords[] = {
      "array",
      "reader",
      "row_indices", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", const_cast<char **>(keywords),
                                   &(PyArray_Type), &array,              // O!
                                   &(PyCapsule_Type), &reader_capsule,   // O!
                                   &(PyArray_Type), &row_indices_array)) // O!
  {
    return nullptr;
  }

  char const *name = PyCapsule_GetName(reader_capsule);
  if (!PyCapsule_IsValid(reader_capsule, name))
  {
    PyErr_SetString(PyExc_ValueError, "Invalid FloatReader capsule");
    return nullptr;
  }
  TSVFloatReader *reader =
      (TSVFloatReader *)PyCapsule_GetPointer(reader_capsule, name);

  /* Get pointers to the arrays */
  if (!CheckFloatArray(array))
  {
    return nullptr;
  }
  int flags = PyArray_FLAGS(array);
  if (!(flags & NPY_ARRAY_WRITEABLE))
  {
    PyErr_SetString(PyExc_ValueError, "`array` must be writable");
    return nullptr;
  }

  auto r = CheckIndexArray(row_indices_array);
  if (!r)
  {
    return nullptr;
  }
  auto [row_indices, row_index_count] = *r;

  size_t column_index_count = reader->column_index_count_;
  if (PyArray_SHAPE(array)[1] != static_cast<npy_intp>(column_index_count) ||
      PyArray_SHAPE(array)[0] != static_cast<npy_intp>(row_index_count))
  {
    PyErr_SetString(PyExc_ValueError,
                    "The shape of `array` does not match the shape of the "
                    "row and column indices");
    return nullptr;
  }

  reader->array_ = array;
  reader->row_indices_ = row_indices;
  reader->row_index_count_ = row_index_count;
  reader->row_indices_index_ = 0;

  Py_BEGIN_ALLOW_THREADS
  try
  {
    reader->loop();
  }
  catch (const std::exception &e)
  {
    Py_BLOCK_THREADS
        PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  Py_END_ALLOW_THREADS

      Py_RETURN_NONE;
}

struct VCFFloatReader : FloatReader<VCFFloatReader>
{
  static constexpr const char *name = "vcf_float_reader";
  VCFFloatReader(PyArrayObject *array,
                 int file_descriptor, size_t skip_bytes,
                 size_t column_count, uint32_t *column_indices,
                 size_t column_index_count, size_t field_index,
                 uint32_t *row_indices, size_t row_index_count,
                 size_t ring_buffer_size)
      : FloatReader(array,
                    file_descriptor, skip_bytes, column_count,
                    column_indices, column_index_count,
                    row_indices, row_index_count,
                    ring_buffer_size),
        field_index_(field_index) {}
  virtual ~VCFFloatReader() = default;

  std::string_view prepare_token(std::string_view token)
  {
    auto begin = token.begin();
    size_t field_index = field_index_;
    while (field_index > 0)
    {
      if (*begin == field_delimiter)
      {
        field_index--;
      }
      begin++;
    }
    return std::string_view(begin, token.end() - begin);
  }

private:
  unsigned int field_index_;
  static constexpr char field_delimiter = ':';
};

static PyObject *CreateVCFFloatReader(PyObject *self, PyObject *args, PyObject *kwargs)
{
  int file_descriptor = 0;
  unsigned long skip_bytes = 0;
  unsigned int column_count = 0;
  PyArrayObject *column_indices_array;
  unsigned int field_index = 0;
  unsigned int ring_buffer_size = default_ring_buffer_size;

  static const char *keywords[] = {
      "file_descriptor",
      "skip_bytes",
      "column_count",
      "column_indices",
      "field_index",
      "ring_buffer_size", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ikIO!I|$I", const_cast<char **>(keywords),
                                   &file_descriptor,                       // i
                                   &skip_bytes,                            // k
                                   &column_count,                          // I
                                   &(PyArray_Type), &column_indices_array, // O!
                                   &field_index,                           // I
                                   &ring_buffer_size))                     // I optional
  {
    return nullptr;
  }

  /* Get pointers to the arrays. */
  auto c = CheckIndexArray(column_indices_array);
  if (!c)
  {
    return nullptr;
  }
  auto [column_indices, column_index_count] = *c;

  VCFFloatReader *reader = nullptr;
  try
  {
    reader = new VCFFloatReader(
        nullptr, file_descriptor, static_cast<size_t>(skip_bytes),
        static_cast<size_t>(column_count), column_indices, column_index_count,
        field_index, nullptr, 0, static_cast<size_t>(ring_buffer_size));
  }
  catch (const std::exception &e)
  {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  if (!reader)
  {
    PyErr_SetString(PyExc_ValueError, "Failure creating FloatReader");
    return nullptr;
  }
  PyObject *capsule =
      PyCapsule_New(reader, VCFFloatReader::name, DestroyFloatReader<VCFFloatReader>);

  if (!capsule)
  {
    PyErr_SetString(PyExc_ValueError, "Failure creating PyCapsule");
    return nullptr;
  }

  return capsule;
}

static PyObject *RunVCFFloatReader(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyArrayObject *array;
  PyObject *reader_capsule;
  PyArrayObject *row_indices_array;

  static const char *keywords[] = {
      "array",
      "reader",
      "row_indices", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", const_cast<char **>(keywords),
                                   &(PyArray_Type), &array,              // O!
                                   &(PyCapsule_Type), &reader_capsule,   // O!
                                   &(PyArray_Type), &row_indices_array)) // O!
  {
    return nullptr;
  }

  char const *name = PyCapsule_GetName(reader_capsule);
  if (!PyCapsule_IsValid(reader_capsule, name))
  {
    PyErr_SetString(PyExc_ValueError, "Invalid FloatReader capsule");
    return nullptr;
  }
  VCFFloatReader *reader =
      (VCFFloatReader *)PyCapsule_GetPointer(reader_capsule, name);

  /* Get pointers to the arrays */
  if (!CheckFloatArray(array))
  {
    return nullptr;
  }
  int flags = PyArray_FLAGS(array);
  if (!(flags & NPY_ARRAY_WRITEABLE))
  {
    PyErr_SetString(PyExc_ValueError, "`array` must be writable");
    return nullptr;
  }

  auto r = CheckIndexArray(row_indices_array);
  if (!r)
  {
    return nullptr;
  }
  auto [row_indices, row_index_count] = *r;

  size_t column_index_count = reader->column_index_count_;
  if (PyArray_SHAPE(array)[1] != static_cast<npy_intp>(column_index_count) ||
      PyArray_SHAPE(array)[0] != static_cast<npy_intp>(row_index_count))
  {
    PyErr_SetString(PyExc_ValueError,
                    "The shape of `array` does not match the shape of the "
                    "row and column indices");
    return nullptr;
  }

  reader->array_ = array;
  reader->row_indices_ = row_indices;
  reader->row_index_count_ = row_index_count;
  reader->row_indices_index_ = 0;

  Py_BEGIN_ALLOW_THREADS
  try
  {
    reader->loop();
  }
  catch (const std::exception &e)
  {
    Py_BLOCK_THREADS
        PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  Py_END_ALLOW_THREADS

      Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {.ml_name = "create_vcf_float_reader",
     .ml_meth = _PyCFunction_CAST(CreateVCFFloatReader),
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "Create a reader object for VCF files"},
    {.ml_name = "run_vcf_float_reader",
     .ml_meth = _PyCFunction_CAST(RunVCFFloatReader),
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "Read fields from specified rows and columns into a 2D array"},
    {.ml_name = "create_tsv_float_reader",
     .ml_meth = _PyCFunction_CAST(CreateTSVFloatReader),
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "Create a reader object for TSV files"},
    {.ml_name = "run_tsv_float_reader",
     .ml_meth = _PyCFunction_CAST(RunTSVFloatReader),
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "Read all specified rows and columns into a 2D array"},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_read_float",
    .m_doc = "A module to read structured data from delimited files",
    /* Setting m_size to -1 means that the module does not support
     * sub-interpreters, because it has global state.
     */
    .m_size = -1,
    .m_methods = methods,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL};

PyMODINIT_FUNC PyInit__read_float(void)
{
  import_array();
  PyObject *module = PyModule_Create(&moduledef);
  return module;
}
