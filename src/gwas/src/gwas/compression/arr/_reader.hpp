#pragma once

#include "_ring_buffer.hpp"

constexpr size_t cache_line_size = 64;

template <class P>
class Reader : RingBuffer
{
public:
    uint32_t *column_indices_;
    size_t column_index_count_;

    void loop();

protected:
    Reader(int file_descriptor, size_t skip_bytes, size_t column_count,
           uint32_t *column_indices, size_t column_index_count,
           size_t ring_buffer_size)
        : RingBuffer(ring_buffer_size),
          column_indices_(column_indices), column_index_count_(column_index_count),
          column_count_(column_count),
          file_descriptor_(file_descriptor), skip_bytes_(skip_bytes)
    {
        read_index_ = skip_bytes_;
        proc_index_ = skip_bytes_;

        if (column_count_ == 0)
        {
            PyErr_SetString(PyExc_ValueError,
                            "`column_count` must be greater than 0");
            throw error_already_set();
        }
        if (column_index_count_ == 0)
        {
            PyErr_SetString(PyExc_ValueError,
                            "`column_index_count` must be greater than 0");
            throw error_already_set();
        }
        if (ring_buffer_size % page_size != 0)
        {
            PyErr_SetString(PyExc_ValueError,
                            "`ring_buffer_size` must be a multiple of page size");
            throw error_already_set();
        }
    }
    virtual ~Reader() = default;

    size_t column_count_;

private:
    int file_descriptor_;
    size_t skip_bytes_;

    void remainder_loop(size_t &start);

    __attribute__((target("default"))) void default_loop();
    __attribute__((target("sse2"))) void sse2_loop();
    __attribute__((target("avx2,bmi"))) void avx2_bmi_loop();
};

template <class P>
void Reader<P>::loop()
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

template <class P>
void Reader<P>::remainder_loop(size_t &start)
{
    constexpr std::array delimiters = P::delimiters;
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
            reinterpret_cast<P *>(this)->on_token(
                get_token(start, read_index_));
            start = read_index_ + 1;
        }
    }
}

template <class P>
void Reader<P>::default_loop()
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

template <class P>
void Reader<P>::sse2_loop()
{
    constexpr std::array delimiters = P::delimiters;
    typedef long long __m128i_a
        __attribute__((__vector_size__(sizeof(__m128i))));
    alignas(cache_line_size) std::array<__m128i_a, delimiters.size()>
        vectorized_delimiters{0};
    std::transform(delimiters.begin(), delimiters.end(),
                   vectorized_delimiters.begin(), _mm_set1_epi8);

    size_t start = read_index_;
    while (push_from_pipe(file_descriptor_))
    {
        if (write_index_ < start)
        {
            continue;
        }

        for (; read_index_ + sizeof(__m128i) - 1 < write_index_;
             read_index_ += sizeof(__m128i))
        {
            __m128i c = get_m128i(read_index_);
            unsigned int is_delimiter;
            for (size_t i = 0; i < vectorized_delimiters.size(); i++)
            {
                unsigned int is_this_delimiter =
                    _mm_movemask_epi8(_mm_cmpeq_epi8(c, vectorized_delimiters[i]));
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

                bool continue_reading = reinterpret_cast<P *>(this)->on_token(
                    get_token(start, read_index_ + index));
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

template <class P>
void Reader<P>::avx2_bmi_loop()
{
    constexpr std::array delimiters = P::delimiters;
    typedef long long __m256i_a
        __attribute__((__vector_size__(sizeof(__m256i))));
    alignas(cache_line_size) std::array<__m256i_a, delimiters.size()>
        vectorized_delimiters{0};
    std::transform(delimiters.begin(), delimiters.end(),
                   vectorized_delimiters.begin(), _mm256_set1_epi8);

    size_t start = read_index_;
    while (push_from_pipe(file_descriptor_))
    {
        if (write_index_ < start)
        {
            continue;
        }

        for (; read_index_ + sizeof(__m256i) - 1 < write_index_;
             read_index_ += sizeof(__m256i))
        {
            __m256i c = get_m256i(read_index_);
            unsigned int is_delimiter;
            for (size_t i = 0; i < vectorized_delimiters.size(); i++)
            {
                unsigned int is_this_delimiter = _mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(c, vectorized_delimiters[i]));
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

                bool continue_reading = reinterpret_cast<P *>(this)->on_token(
                    get_token(start, read_index_ + index));
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
