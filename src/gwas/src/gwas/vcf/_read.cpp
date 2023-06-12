#include <algorithm> // std::copy
#include <array>     // std::array
#include <charconv>  // std::from_chars
#include <cstdint>   // uint32_t
#include <exception> // std::exception
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

const size_t page_size = sysconf(_SC_PAGESIZE);
const size_t level1_cache_size = sysconf(_SC_LEVEL1_DCACHE_SIZE);

constexpr size_t cache_line_size = 64;

class error_already_set : public std::exception
{
};

struct RingBuffer
{
    std::vector<char> data_;
    size_t read_index_{0};
    size_t proc_index_{0};
    size_t write_index_{0};
    std::vector<char> value_buffer_;

    size_t push_from_pipe(const int file_descriptor)
    {
        const size_t index = std::min(read_index_, proc_index_);

        const size_t write_loop_count = write_index_ / data_.size();
        const size_t loop_count = index / data_.size();

        // std::cout << "push_from_pipe write_loop_count " << write_loop_count << std::endl;
        // std::cout << "push_from_pipe loop_count " << loop_count << std::endl;

        const size_t actual_write_index = write_index_ % data_.size();
        const size_t actual_index = index % data_.size();

        const size_t read_size = (write_loop_count == loop_count)
                                     ? data_.size() - actual_write_index
                                     : actual_index - actual_write_index;

        // std::cout << "push_from_pipe write_index_ " << write_index_ << std::endl;
        // std::cout << "push_from_pipe index " << index << std::endl;
        // std::cout << "push_from_pipe actual_write_index " << actual_write_index << std::endl;
        // std::cout << "push_from_pipe actual_index " << actual_index << std::endl;
        // std::cout << "push_from_pipe data_.size " << data_.size() << std::endl;
        // std::cout << "push_from_pipe read_size " << read_size << std::endl;

        /* Read as much as we can from the pipe. */
        const ssize_t bytes_read = read(
            file_descriptor,
            data_.data() + actual_write_index,
            read_size);
        if (bytes_read < 0)
        {
            std::cerr << "Error reading from file descriptor" << std::endl;
            PyErr_SetString(PyExc_IOError, "Error reading from file descriptor");
            throw error_already_set();
        }

        // std::cout << "push_from_pipe bytes_read " << bytes_read << std::endl;
        // std::cout << "push_from_pipe data_ \"" << std::string_view(data_.data() + actual_write_index, bytes_read) << "\"" << std::endl;

        write_index_ += bytes_read;

        return bytes_read;
    }

    char get_byte(const size_t index)
    {
        size_t actual_index = index % data_.size();
        return data_[actual_index];
    }

    __attribute__((target("sse2"))) __m128i get_m128i(const size_t index)
    {
        size_t first_index = index;
        size_t last_index = index + sizeof(__m128i) - 1;

        size_t actual_first_index = first_index % data_.size();
        size_t actual_last_index = last_index % data_.size();

        if (actual_first_index < actual_last_index) [[likely]]
        {
            return _mm_loadu_si128(reinterpret_cast<__m128i *>(data_.data() + actual_first_index));
        }
        else
        {
            std::array<char, sizeof(__m256i)> c = {0};
            std::copy(
                data_.begin() + actual_first_index,
                data_.end(),
                c.begin());
            std::copy(
                data_.begin(),
                data_.begin() + actual_last_index + 1,
                c.begin() + (data_.size() - actual_first_index));
            return _mm_set_epi8(
                c[15], c[14], c[13], c[12], c[11], c[10], c[9], c[8], c[7], c[6], c[5], c[4], c[3], c[2], c[1], c[0]);
        }
    }

    __attribute__((target("avx2"))) __m256i get_m256i(const size_t index)
    {
        size_t first_index = index;
        size_t last_index = index + sizeof(__m256i) - 1;

        size_t actual_first_index = first_index % data_.size();
        size_t actual_last_index = last_index % data_.size();

        if (actual_first_index < actual_last_index) [[likely]]
        {
            // std::cout << "get_m256i "
            //           << "index " << index << " "
            //           << "first_index " << first_index << " "
            //           << "last_index " << last_index << " "
            //           << "actual_first_index " << actual_first_index << " "
            //           << "actual_last_index " << actual_last_index << " "
            //           << "data_.size() " << data_.size() << " "
            //           << "\"" << std::string_view(data_.data() + actual_first_index, sizeof(__m256i)) << "\""
            //           << std::endl;

            return _mm256_loadu_si256(reinterpret_cast<__m256i *>(data_.data() + actual_first_index));
        }
        else
        {
            std::array<char, sizeof(__m256i)> c = {0};
            // for (size_t i = 0; i < sizeof(__m256i); i++)
            // {
            //     c[i] = get_byte(index + i);
            // }
            std::copy(
                data_.begin() + actual_first_index,
                data_.end(), c.begin());
            std::copy(
                data_.begin(),
                data_.begin() + actual_last_index + 1,
                c.begin() + (data_.size() - actual_first_index));
            // std::cout << "get_m256i "
            //           << "index " << index << " "
            //           << "first_index " << first_index << " "
            //           << "last_index " << last_index << " "
            //           << "actual_first_index " << actual_first_index << " "
            //           << "actual_last_index " << actual_last_index << " "
            //           << "data_.size() " << data_.size() << " "
            //           << "\"" << std::string_view(c.data(), c.size()) << "\""
            //           << std::endl;
            return _mm256_set_epi8(
                c[31], c[30], c[29], c[28], c[27], c[26], c[25], c[24], c[23], c[22], c[21], c[20], c[19], c[18], c[17], c[16],
                c[15], c[14], c[13], c[12], c[11], c[10], c[9], c[8], c[7], c[6], c[5], c[4], c[3], c[2], c[1], c[0]);
        }
    }

    std::string_view get_token(const size_t begin_index, const size_t end_index)
    {
        const unsigned int size = end_index - begin_index;

        const size_t actual_begin_index = begin_index % data_.size();
        const size_t actual_end_index = end_index % data_.size();

        std::string_view token;

        if (actual_begin_index < actual_end_index) [[likely]]
        {
            token = std::string_view(data_.data() + actual_begin_index, size);
        }
        else
        {
            value_buffer_.clear();
            value_buffer_.reserve(size);
            value_buffer_.insert(
                value_buffer_.end(),
                data_.begin() + actual_begin_index,
                data_.end());
            value_buffer_.insert(
                value_buffer_.end(),
                data_.begin(),
                data_.begin() + actual_end_index);

            // std::cout << "wrap_token"
            //           << " \"" << std::string_view(data_.begin() + actual_begin_index, data_.end()) << "\""
            //           << " \"" << std::string_view(data_.begin(), data_.begin() + actual_end_index) << "\""
            //           << " \"" << std::string_view(value_buffer_.data(), value_buffer_.size()) << "\""
            //           << std::endl;

            token = std::string_view(value_buffer_.data(), size);
        }

        // std::cout << "get_token"
        //           << " begin_index " << static_cast<int>(begin_index)
        //           << " end_index " << static_cast<int>(end_index)
        //           << " actual_begin_index " << static_cast<int>(actual_begin_index)
        //           << " actual_end_index " << static_cast<int>(actual_end_index)
        //           << " value_buffer_.size() " << static_cast<int>(value_buffer_.size())
        //           << " size " << static_cast<int>(size)
        //           << " " << token << std::endl;

        return token;
    }

    RingBuffer(size_t size) : data_(size)
    {
        if (size % page_size != 0)
        {
            std::cerr << "`RingBuffer` size must be a multiple of page size" << std::endl;
            PyErr_SetString(PyExc_ValueError, "`RingBuffer` size must be a multiple of page size");
            throw error_already_set();
        }
    }
    virtual ~RingBuffer() = default;
};

template <class Parser>
struct Reader : RingBuffer
{
    int file_descriptor_;

    size_t skip_bytes_;

    size_t column_count_;
    uint32_t *column_indices_;
    size_t column_index_count_;

    Reader(
        int file_descriptor,
        size_t skip_bytes,
        size_t column_count,
        uint32_t *column_indices,
        size_t column_index_count,
        size_t ring_buffer_size = level1_cache_size)
        : RingBuffer(ring_buffer_size),
          file_descriptor_(file_descriptor),
          skip_bytes_(skip_bytes),
          column_count_(column_count),
          column_indices_(column_indices),
          column_index_count_(column_index_count)
    {
        read_index_ = skip_bytes_;

        if (column_count_ == 0)
        {
            PyErr_SetString(PyExc_ValueError, "`column_count` must be greater than 0");
            throw error_already_set();
        }
        if (column_index_count_ == 0)
        {
            PyErr_SetString(PyExc_ValueError, "`column_index_count` must be greater than 0");
            throw error_already_set();
        }
        if (ring_buffer_size % page_size != 0)
        {
            PyErr_SetString(PyExc_ValueError, "`ring_buffer_size` must be a multiple of page size");
            throw error_already_set();
        }
    }
    virtual ~Reader() = default;

    void loop()
    {
        if (__builtin_cpu_supports("bmi"))
        {
            if (__builtin_cpu_supports("avx2"))
            {
                return avx2_bmi_loop();
            }
        }
        if (__builtin_cpu_supports("sse2"))
        {
            return sse2_loop();
        }
        return default_loop();
    }

    void remainder_loop(size_t &start)
    {
        constexpr std::array delimiters = Parser::delimiters;
        for (; read_index_ < write_index_; read_index_++)
        {
            char c = get_byte(read_index_);
            bool is_delimiter = false;
            for (size_t i = 0; i < delimiters.size(); i++)
            {
                if (c == delimiters[i])
                {
                    is_delimiter = true;
                    break;
                }
            }
            if (is_delimiter)
            {
                reinterpret_cast<Parser *>(this)->on_token(get_token(start, read_index_));
                start = read_index_ + 1;
            }
        }
    }

    __attribute__((target("default"))) void default_loop()
    {
        size_t start = skip_bytes_;
        read_index_ = skip_bytes_;
        while (push_from_pipe(file_descriptor_))
        {
            if (write_index_ < start)
            {
                continue;
            }
            remainder_loop(start);
        }
    }

    __attribute__((target("sse2"))) void sse2_loop()
    {
        constexpr std::array delimiters = Parser::delimiters;
        typedef long long __m128i_a __attribute__((__vector_size__(sizeof(__m128i))));
        alignas(cache_line_size) std::array<__m128i_a, delimiters.size()> vectorized_delimiters{0};
        std::transform(delimiters.begin(), delimiters.end(),
                       vectorized_delimiters.begin(), _mm_set1_epi8);

        size_t start = read_index_;
        while (push_from_pipe(file_descriptor_))
        {
            if (write_index_ < start)
            {
                continue;
            }

            for (; read_index_ + sizeof(__m128i) - 1 < write_index_; read_index_ += sizeof(__m128i))
            {
                __m128i c = get_m128i(read_index_);
                unsigned int is_delimiter;
                for (size_t i = 0; i < vectorized_delimiters.size(); i++)
                {
                    unsigned int is_this_delimiter = _mm_movemask_epi8(_mm_cmpeq_epi8(c, vectorized_delimiters[i]));
                    if (i == 0)
                    {
                        is_delimiter = is_this_delimiter;
                    }
                    else
                    {
                        is_delimiter |= is_this_delimiter;
                    }
                }

                while (is_delimiter)
                {
                    size_t index = __builtin_ctz(is_delimiter);

                    bool continue_reading = reinterpret_cast<Parser *>(this)->on_token(get_token(start, read_index_ + index));
                    start = read_index_ + index + 1;

                    if (!continue_reading)
                    {
                        read_index_ = start;
                        return;
                    }

                    is_delimiter &= ~(1 << index);
                }
                proc_index_ = start;
            }
        }
        remainder_loop(start);
    }

    __attribute__((target("avx2,bmi"))) void avx2_bmi_loop()
    {
        constexpr std::array delimiters = Parser::delimiters;
        typedef long long __m256i_a __attribute__((__vector_size__(sizeof(__m256i))));
        alignas(cache_line_size) std::array<__m256i_a, delimiters.size()> vectorized_delimiters{0};
        std::transform(delimiters.begin(), delimiters.end(),
                       vectorized_delimiters.begin(), _mm256_set1_epi8);

        size_t start = read_index_;
        while (push_from_pipe(file_descriptor_))
        {
            if (write_index_ < start)
            {
                continue;
            }

            for (; read_index_ + sizeof(__m256i) - 1 < write_index_; read_index_ += sizeof(__m256i))
            {
                __m256i c = get_m256i(read_index_);
                unsigned int is_delimiter;
                for (size_t i = 0; i < vectorized_delimiters.size(); i++)
                {
                    unsigned int is_this_delimiter = _mm256_movemask_epi8(_mm256_cmpeq_epi8(c, vectorized_delimiters[i]));
                    if (i == 0)
                    {
                        is_delimiter = is_this_delimiter;
                    }
                    else
                    {
                        is_delimiter |= is_this_delimiter;
                    }
                }

                while (is_delimiter)
                {
                    size_t index = __builtin_ctz(is_delimiter);

                    bool continue_reading = reinterpret_cast<Parser *>(this)->on_token(get_token(start, read_index_ + index));
                    start = read_index_ + index + 1;

                    if (!continue_reading)
                    {
                        read_index_ = start;
                        return;
                    }

                    is_delimiter &= ~(1 << index);
                }
                proc_index_ = start;
            }
        }
        remainder_loop(start);
    }
};

struct StrReader : public Reader<StrReader>
{
    std::vector<PyObject *> tokens_;
    PyObject *parser_;

    PyObject *rows_;

    size_t column_indices_index_{0};
    size_t token_index_{0};

    static constexpr std::array<char, 2> delimiters = {'\n', '\t'};

    bool on_token(std::string_view token)
    {
        const size_t row_index = token_index_ / column_count_;
        const size_t column_index = token_index_ % column_count_;
        token_index_++;

        // std::cout << "on_token"
        //           << " row_index " << static_cast<int>(row_index)
        //           << " column_index " << static_cast<int>(column_index)
        //           << " column_indices_index_ " << static_cast<int>(column_indices_index_)
        //           << " column_indices_[column_indices_index_] " << static_cast<int>(column_indices_[column_indices_index_])
        //           << " " << token << std::endl;

        if (column_indices_[column_indices_index_] == column_index)
        {
            PyObject *py_token =
                PyUnicode_DecodeASCII(token.data(), token.size(), "ignore");
            tokens_[column_indices_index_] = py_token;
            column_indices_index_++;

            if (column_indices_index_ == column_index_count_)
            {
                /* We have finished reading a record and can move on to the next. */
                PyObject *result = PyObject_Vectorcall(parser_, tokens_.data(), tokens_.size(), nullptr);
                if (!result)
                {
                    throw error_already_set();
                }
                Py_INCREF(result);
                for (PyObject *py_token : tokens_)
                {
                    Py_DECREF(py_token);
                }

                PyList_Append(rows_, result);
                Py_DECREF(result);

                column_indices_index_ = 0;
            }
        }
        return true;
    }

    StrReader(
        PyObject *rows,
        PyObject *parser,
        int file_descriptor,
        size_t skip_bytes,
        size_t column_count,
        uint32_t *column_indices,
        size_t column_index_count,
        size_t ring_buffer_size = level1_cache_size)
        : Reader(file_descriptor,
                 skip_bytes,
                 column_count,
                 column_indices,
                 column_index_count,
                 ring_buffer_size),
          tokens_(column_index_count),
          parser_(parser),
          rows_(rows)
    {
    }
    virtual ~StrReader() = default;
};

struct FloatReader : public Reader<FloatReader>
{
    double *float_array_;

    unsigned int field_index_;

    uint32_t *row_indices_;
    size_t row_index_count_;

    size_t token_index_{0};
    size_t column_indices_index_{0};
    size_t row_indices_index_{0};

    static constexpr char field_delimiter = ':';
    static constexpr std::array<char, 3> delimiters = {'\n', '\t'};

    static constexpr const char *name = "float_reader";

    bool on_token(std::string_view token)
    {
        const unsigned int current_row_index = token_index_ / column_count_;
        const unsigned int current_column_index = token_index_ % column_count_;

        /* Always increment the token counter. */
        token_index_++;

        // py::str record_str = py::reinterpret_steal<py::str>(
        //     PyUnicode_DecodeASCII(token.data, token.size, nullptr));
        // py::print(
        //     "on_token:",
        //     current_row_index, current_column_index,
        //     row_indices_index_, column_indices_index_,
        //     row_indices_[row_indices_index_], column_indices_[column_indices_index_],
        //     record_str,
        //     column_count_);

        if (row_indices_[row_indices_index_] == current_row_index)
        {
            if (column_indices_[column_indices_index_] == current_column_index)
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

                size_t float_index = column_indices_index_ + row_indices_index_ * column_index_count_;
                if (std::from_chars(begin, token.end(), float_array_[float_index]).ec != std::errc{})
                {
                    std::cerr << "Could not parse float " << token << std::endl;
                    PyErr_SetString(PyExc_ValueError, "Could not parse float");
                    throw error_already_set();
                }

                // py::print("from_chars", column_indices_index_, row_indices_index_, column_index_count_, float_index, float_array_[float_index]);

                column_indices_index_++;

                if (column_indices_index_ == column_index_count_)
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

    FloatReader(
        double *float_array,
        int file_descriptor,
        size_t skip_bytes,
        size_t column_count,
        uint32_t *column_indices,
        size_t column_index_count,
        size_t field_index,
        uint32_t *row_indices,
        size_t row_index_count,
        size_t ring_buffer_size = level1_cache_size)
        : Reader(file_descriptor, skip_bytes, column_count, column_indices, column_index_count, ring_buffer_size),
          float_array_(float_array),
          field_index_(field_index),
          row_indices_(row_indices), row_index_count_(row_index_count)
    {
    }
    virtual ~FloatReader() = default;
};

static bool CheckIndexArray(PyArrayObject *array)
{
    if (PyArray_NDIM(array) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Index arrays must be a one-dimensional");
        return false;
    }

    if (PyArray_TYPE(array) != NPY_UINT32)
    {
        PyErr_SetString(PyExc_ValueError, "Index arrays must be an array of unsigned integers");
        return false;
    }

    if (PyArray_STRIDES(array)[0] != sizeof(unsigned int))
    {
        PyErr_SetString(PyExc_ValueError, "Index arrays must be a contiguous array");
        return false;
    }

    return true;
}

static PyObject *ReadStr(
    PyObject * /* self */,
    PyObject *arguments,
    Py_ssize_t argument_count)
{
    PyObject *row_list;
    PyObject *row_parser;
    int file_descriptor = 0;
    unsigned long skip_bytes = 0;
    unsigned int column_count = 0;
    PyArrayObject *column_indices_array;
    if (!PyArg_ParseTuple(
            arguments, "O!OikIO!",
            &(PyList_Type), &row_list,               // O!
            &row_parser,                             // O
            &file_descriptor,                        // i
            &skip_bytes,                             // k
            &column_count,                           // I
            &(PyArray_Type), &column_indices_array)) // O!
    {
        return nullptr;
    }

    if (!CheckIndexArray(column_indices_array))
    {
        return nullptr;
    }

    /* Get pointers to the arrays. */
    size_t column_index_count = PyArray_SIZE(column_indices_array);
    uint32_t *column_indices = reinterpret_cast<uint32_t *>(PyArray_DATA(column_indices_array));

    try
    {
        StrReader reader{
            row_list,
            row_parser,
            file_descriptor,
            static_cast<size_t>(skip_bytes),
            static_cast<size_t>(column_count),
            column_indices, column_index_count};

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

PyCapsule_Destructor DestroyFloatReader = [](PyObject *float_reader_capsule)
{
    if (!PyCapsule_IsValid(float_reader_capsule, FloatReader::name))
    {
        return;
    }

    FloatReader *float_reader = (FloatReader *)PyCapsule_GetPointer(float_reader_capsule, FloatReader::name);
    if (float_reader)
    {
        delete float_reader;
    }
};

static PyObject *
CreateFloatReader(PyObject * /* self */, PyObject *arguments, Py_ssize_t argument_count)
{
    int file_descriptor = 0;
    unsigned long skip_bytes = 0;
    unsigned int column_count = 0;
    PyArrayObject *column_indices_array;
    unsigned int field_index = 0;
    if (!PyArg_ParseTuple(
            arguments, "ikIO!I",
            &file_descriptor,                       // i
            &skip_bytes,                            // k
            &column_count,                          // I
            &(PyArray_Type), &column_indices_array, // O!
            &field_index))                          // I
    {
        return nullptr;
    }

    /* Get pointers to the arrays. */
    size_t column_index_count = PyArray_SIZE(column_indices_array);
    uint32_t *column_indices = reinterpret_cast<uint32_t *>(PyArray_DATA(column_indices_array));

    if (!CheckIndexArray(column_indices_array))
    {
        return nullptr;
    }

    FloatReader *float_reader = nullptr;
    try
    {
        float_reader = new FloatReader(
            nullptr,
            file_descriptor,
            static_cast<size_t>(skip_bytes),
            static_cast<size_t>(column_count),
            column_indices, column_index_count,
            field_index,
            nullptr, 0);
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

    if (!float_reader)
    {
        PyErr_SetString(PyExc_ValueError, "Failure creating FloatReader");
        return nullptr;
    }
    PyObject *float_reader_capsule = PyCapsule_New(
        float_reader,
        FloatReader::name,
        DestroyFloatReader);

    if (!float_reader_capsule)
    {
        PyErr_SetString(PyExc_ValueError, "Failure creating PyCapsule");
        return nullptr;
    }

    return float_reader_capsule;
}

static PyObject *RunFloatReader(
    PyObject * /* self */,
    PyObject *arguments,
    Py_ssize_t argument_count)
{
    PyArrayObject *data_array;
    PyObject *float_reader_capsule;
    PyArrayObject *row_indices_array;
    if (!PyArg_ParseTuple(
            arguments, "O!O!O!",
            &(PyArray_Type), &data_array,             // O!
            &(PyCapsule_Type), &float_reader_capsule, // O!
            &(PyArray_Type), &row_indices_array))     // O!
    {
        return nullptr;
    }

    char const *name = PyCapsule_GetName(float_reader_capsule);
    if (!PyCapsule_IsValid(float_reader_capsule, name))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid FloatReader capsule");
        return nullptr;
    }
    FloatReader *float_reader = (FloatReader *)PyCapsule_GetPointer(float_reader_capsule, name);

    double *data = reinterpret_cast<double *>(PyArray_DATA(data_array));
    if (PyArray_NDIM(data_array) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "`data_array` must be a 2D array");
        return nullptr;
    }
    if (PyArray_TYPE(data_array) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "`data_array` must be an array of doubles");
        return nullptr;
    }
    size_t column_index_count = float_reader->column_index_count_;
    if (PyArray_STRIDES(data_array)[0] != static_cast<npy_intp>(sizeof(double)) ||
        PyArray_STRIDES(data_array)[1] != static_cast<npy_intp>(sizeof(double) * column_index_count))
    {
        PyErr_SetString(PyExc_ValueError, "The strides of `data_array` are not in column-major order");
        return nullptr;
    }

    size_t row_index_count = PyArray_SIZE(row_indices_array);
    uint32_t *row_indices = reinterpret_cast<uint32_t *>(PyArray_DATA(row_indices_array));
    if (!CheckIndexArray(row_indices_array))
    {
        return nullptr;
    }
    if (PyArray_SHAPE(data_array)[0] != static_cast<npy_intp>(column_index_count) ||
        PyArray_SHAPE(data_array)[1] != static_cast<npy_intp>(row_index_count))
    {
        PyErr_SetString(PyExc_ValueError, "The shape of `data_array` does not match the shape of the row and column indices");
        return nullptr;
    }

    float_reader->float_array_ = data;
    float_reader->row_indices_ = row_indices;
    float_reader->row_index_count_ = row_index_count;
    float_reader->row_indices_index_ = 0;
    float_reader->loop();

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {.ml_name = "read_str",
     .ml_meth = _PyCFunction_CAST(ReadStr),
     .ml_flags = METH_VARARGS,
     .ml_doc = "Read all specified columns from a tab-delimited file into a list of records"},
    {.ml_name = "create_float_reader",
     .ml_meth = _PyCFunction_CAST(CreateFloatReader),
     .ml_flags = METH_VARARGS,
     .ml_doc = "Create a reader object"},
    {.ml_name = "run_float_reader",
     .ml_meth = _PyCFunction_CAST(RunFloatReader),
     .ml_flags = METH_VARARGS,
     .ml_doc = "Read all specified rows and columns into a 2D array"},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_read",
    .m_doc = "A module to read structured data from delimited files",
    /* Setting m_size to -1 means that the module does not support sub-interpreters,
     * because it has global state.
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
