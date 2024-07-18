#pragma once

#include <algorithm>   // std::copy
#include <exception>   // std::exception
#include <format>      // std::format
#include <string_view> // std::string_view
#include <vector>      // std::vector

// #include <unistd.h>
#include <x86intrin.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

const size_t page_size = sysconf(_SC_PAGESIZE);
const size_t default_ring_buffer_size = sysconf(_SC_LEVEL2_CACHE_SIZE);

class RingBuffer
{
protected:
    RingBuffer(size_t size) : data_(size)
    {
        if (size % page_size != 0)
        {
            throw std::invalid_argument("`RingBuffer` size must be a multiple of page size");
        }
    }
    virtual ~RingBuffer() = default;

    size_t read_index_{0};
    size_t proc_index_{0};
    size_t write_index_{0};

    size_t push_from_pipe(const int file_descriptor)
    {
        const size_t index = std::min(read_index_, proc_index_);
        const size_t write_loop_count = write_index_ / data_.size();

        /* Counts how many times we have looped around the ring buffer */
        const size_t loop_count = index / data_.size();

        const size_t actual_write_index = write_index_ % data_.size();
        const size_t actual_index = index % data_.size();

        const bool can_overwrite_from_start = write_loop_count <= loop_count;
        const size_t read_size = can_overwrite_from_start
                                     ? data_.size() - actual_write_index
                                     : actual_index - actual_write_index;

        if (read_size == 0) [[unlikely]]
        {
            auto error_string = std::format(
                "Read length is zero for ring buffer of size {} at index {}, loop "
                "count {}, write index {}, write loop count {}, actual index {},  "
                "actual write index {}, and can_overwrite_from_start {}",
                data_.size(), index, loop_count, write_index_, write_loop_count,
                actual_index, actual_write_index, can_overwrite_from_start);
            throw std::runtime_error(error_string);
        }

        /* Read as much as we can from the pipe. */
        const ssize_t bytes_read =
            read(file_descriptor, data_.data() + actual_write_index, read_size);
        if (bytes_read < 0) [[unlikely]]
        {
            throw std::runtime_error("Error reading from file descriptor");
        }

        // std::cout << "push_from_pipe bytes_read " << bytes_read << std::endl;

        write_index_ += bytes_read;

        return bytes_read;
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
            value_buffer_.insert(value_buffer_.end(),
                                 data_.begin() + actual_begin_index, data_.end());
            value_buffer_.insert(value_buffer_.end(), data_.begin(),
                                 data_.begin() + actual_end_index);

            token = std::string_view(value_buffer_.data(), size);
        }

        return token;
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
            return _mm_loadu_si128(
                reinterpret_cast<__m128i *>(data_.data() + actual_first_index));
        }
        else
        {
            std::array<char, sizeof(__m256i)> c = {0};
            std::copy(data_.begin() + actual_first_index, data_.end(), c.begin());
            std::copy(data_.begin(), data_.begin() + actual_last_index + 1,
                      c.begin() + (data_.size() - actual_first_index));
            return _mm_set_epi8(c[15], c[14], c[13], c[12], c[11], c[10], c[9], c[8],
                                c[7], c[6], c[5], c[4], c[3], c[2], c[1], c[0]);
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
            return _mm256_loadu_si256(
                reinterpret_cast<__m256i *>(data_.data() + actual_first_index));
        }
        else
        {
            std::array<char, sizeof(__m256i)> c = {0};

            std::copy(data_.begin() + actual_first_index, data_.end(), c.begin());
            std::copy(data_.begin(), data_.begin() + actual_last_index + 1,
                      c.begin() + (data_.size() - actual_first_index));

            return _mm256_set_epi8(c[31], c[30], c[29], c[28], c[27], c[26], c[25],
                                   c[24], c[23], c[22], c[21], c[20], c[19], c[18],
                                   c[17], c[16], c[15], c[14], c[13], c[12], c[11],
                                   c[10], c[9], c[8], c[7], c[6], c[5], c[4], c[3],
                                   c[2], c[1], c[0]);
        }
    }

private:
    std::vector<char> data_;
    std::vector<char> value_buffer_;
};
